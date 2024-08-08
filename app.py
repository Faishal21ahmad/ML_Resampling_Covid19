import json
import numpy as np
import pandas as pd
import csv
import pickle
from numpy import False_
import matplotlib.pyplot as plt
import os

from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score
from imblearn.metrics import classification_report_imbalanced
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SECRET_KEY"] = "AcrMlCvd19"

# ================ Path Apps ======================== #
pth = '/home/ishal/Documents/Repo/ML_Resampling_Cls_Covid'

path_input = pth + '/data_input/'
path_label = pth + '/label_list/'
path_model = pth + '/model_training/'
path_result = pth + '/result/'

# ================ extensi permis ================= #
ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ================ Cari data ================= #
def caridata(array, target):
    for i, elemen in enumerate(array):
        if elemen == target:
            return 1
    return 0

# ================ Cari data ================= #
def scan_files(folder_path,ext):
    file_info_list = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath) and filename.endswith(ext):
            file_info = {
                "filename": filename
            }
            file_info_list.append(file_info)
    return file_info_list

# ================ ROOT Evaluasi ================= #

@app.route('/', methods=['GET', 'POST'])
def root():
    title = "Evaluasi"
    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            session.pop('filename', None)
            file = request.files['file']
            num_rows = int(request.form['num_rows'])
            session["num_rows"] = num_rows
            
            if 'file' not in request.files or not request.files['file']:
                return render_template('evaluasi.html', title=title, fil=0)

            if file and allowed_file(file.filename):
                try:
                    # Baca file CSV
                    df = pd.read_csv(file)

                    # Batasi jumlah baris
                    limited_df = df.head(num_rows)
                    
                    filename = secure_filename(file.filename)
                    filename_no_ext, ext = os.path.splitext(filename)
                    new_filename = f'{num_rows}_{filename.split(".")[0]}{ext}'
                    
                    # Simpan file CSV yang sudah dibatasi ke folder data_input
                    data_folder = os.path.join(app.root_path, 'data_input')
                    if not os.path.exists(data_folder):
                        os.makedirs(data_folder)

                    path = os.path.join(data_folder, new_filename)
                    limited_df.to_csv(path, index=False)
                    
                    session["filename"] = new_filename
                except Exception as e:
                    ms = "Format File Tidak didukung"
                    return render_template('evaluasi.html', title=title, error=ms)

            try:
                pat = "data_input/" + new_filename
                df = pd.read_csv(pat)
                count_data = len(df)
                session["count_data"] = count_data
                co = []
                for item in df:
                    co.append(item)

                data = []
                with open(pat, "r") as f:
                    content = f.readlines()
                    for line in content[0:]:
                        columns = line.strip().split(",")
                        h = np.count_nonzero(columns)
                        if h == False:
                            i = 0
                        else:
                            i = h
                        data.append(columns)

                session["colLab"] = co

                # retrun if post number 1
                return render_template('evaluasi.html', title=title, fil=new_filename, data=data, co=co, i=i, count_data=count_data)
            except Exception as e:
                ms = "Format File Tidak didukung"
                return render_template('evaluasi.html', title=title, error=ms)

        if proses == "2":
            cls = request.form.get('kls')

            if 'kls' not in request.form or not request.form.get('kls'):
                title = "Evaluasi"
                return render_template('evaluasi.html', title=title, cls=0, fil=0)

            co = session["colLab"]
            fil = session["filename"]
            num_rows = session["num_rows"]
            count_data = session["count_data"]

            df = pd.read_csv("data_input/" + fil)
            fit = [value for value in co if value not in cls]
            fil = fil[:-4] if fil.endswith('.csv') else fil
            

            pickle.dump(fit, open(f"" + path_label + fil + ".pkl", "wb"))

            data = []
            with open("data_input/" + fil + ".csv", "r") as f:
                content = f.readlines()
                for line in content[0:]:
                    columns = line.strip().split(",")
                    i = np.count_nonzero(columns)
                    data.append(columns)

            X = df.loc[:, fit].values
            y = df.loc[:, cls].values

            # resampling proses 
            try:
                smote = SMOTE(random_state=50)
                adasyn = ADASYN(random_state=50)
                X_res_smote, y_res_smote = smote.fit_resample(X, y)
                X_res_adasyn, y_res_adasyn = adasyn.fit_resample(X, y)
            except Exception as e:
                os.remove("data_input/" + fil + ".csv")
                os.remove("label_list/" + fil + ".pkl")
                ms = "Sample data " + str(num_rows) + " tidak dapat digunakan"
                return render_template('evaluasi.html', title=title, error=ms)

            # insert result resampling in array 
            resamplings = [
                ('Oriset', y, X),
                ('Smote', y_res_smote, X_res_smote),
                ('Adasyn', y_res_adasyn, X_res_adasyn)
            ]
            
            # Count label 
            resamcount = [[rep[0], Counter(rep[1])] for rep in resamplings]
            
            # inisialisasi test split 
            tss = [
                ('TS_9:1', 0.1),
                ('TS_8:2', 0.2),
                ('TS_7:3', 0.3)
            ]
            # Spliting data 
            test_trains = [[rep[0], ts[0], train_test_split(rep[2], rep[1], test_size=ts[1], random_state=50)] for ts in tss for rep in resamplings]

            # inisialisasi model 
            models = [
                ('SVM', SVC()),
                ('RF', RandomForestClassifier()),
                ('NN', MLPClassifier())
            ]
            
            # Model Training 
            model_trains = [[model[0], test_train[0], test_train[1], test_train[2][1], test_train[2][3], model[1].fit(test_train[2][0], test_train[2][2])] for model in models for test_train in test_trains]

            # Export Model for prediction
            model_tok = [[f"{model[0]}_{model[1]}_{model[2]}", model[5]] for model in model_trains]
            pickle.dump(model_tok, open(f"{path_model}{fil}.pkl", "wb"))

            # Model prediction 
            preds = [[model_train[0], model_train[1], model_train[2], model_train[4], model_train[5].predict(model_train[3])] for model_train in model_trains]

            # Evaluation 
            result = []
            for pred in preds:
                acr = round(accuracy_score(pred[3], pred[4])*100, 2)
                report = classification_report_imbalanced(pred[3], pred[4], output_dict=True)
                pre = round(report['avg_pre']*100, 2)
                rec = round(report['avg_rec']*100, 2)
                spe = round(report['avg_spe']*100, 2)
                f1 = round(report['avg_f1']*100, 2)
                geo = round(report['avg_geo']*100, 2)
                result.append([pred[0], pred[1], pred[2], acr, pre, rec, spe, f1, geo])

            grub = [[fil, resamcount, result]]
            pickle.dump(grub, open(f"{path_result}{fil}.pkl", "wb"))

            # retrun if post number 2
            return render_template('evaluasi.html', title=title, data=data, i=i, co=co, resamcount=resamcount, result=result, fil=fil, cls=cls, fit=fit, count_data=count_data)

        # retrun if post not proses
        return render_template('evaluasi.html', title=title)
    session.pop('filename', None)
    # retrun if get
    return render_template('evaluasi.html', title=title)


@app.route('/banding', methods=['GET', 'POST'])
def banding():
    title = "Banding"
    file = scan_files(path_result,".pkl")

    fis = []
    for fil in file:
        fis.append(fil["filename"])

    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            str_file = [str(x)for x in request.form.values()]
            str_file = np.delete(str_file, -1)
            str_file = str_file.tolist()
            session["str_file"] = str_file

            grub = []
            for name in str_file:
                grub.append(pickle.load(open('result/' + name, 'rb')))

            alg = []
            rs = []
            ts = []

            if alg == []:
                alg = ['SVM', 'RF', 'NN']

            if rs == []:
                rs = ['Oriset', 'Smote', 'Adasyn']

            if ts == []:
                ts = ['TS_9:1', 'TS_8:2', 'TS_7:3']

            # return render_template('banding.html', title=title, file=fis, filed=str_file, grub="none")
            return render_template('banding.html', title=title, file=fis, filed=str_file, grub=grub, rs=rs, alg=alg, ts=ts)
        if proses == "2":
            file = scan_files(path_result,".pkl")

            if "str_file" not in session:
                session["str_file"] = []

            str_file = session["str_file"]

            files = []
            for fil in file:
                files.append(fil["filename"])

            grub = []
            for name in str_file:
                grub.append(pickle.load(open('result/' + name, 'rb')))

            alg = [str(x) for x in request.form.getlist('algo')]
            rs = [str(x) for x in request.form.getlist('resam')]
            ts = [str(x) for x in request.form.getlist('dts')]

            if alg == []:
                alg = ['SVM', 'RF', 'NN']

            if rs == []:
                rs = ['Oriset', 'Smote', 'Adasyn']

            if ts == []:
                ts = ts = ['TS_9:1', 'TS_8:2', 'TS_7:3']
            return render_template('banding.html', title=title, filed=str_file, file=files, grub=grub, rs=rs, alg=alg, ts=ts)

    return render_template('banding.html', title=title, file=fis, grub="none")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    title = "Prediksi"
    file_info_list = scan_files(path_model,".pkl")

    if request.method == 'POST':
        proses = request.form.get('proses')
        if proses == "1":
            sel = request.form['sel']
            session["sel"] = sel
            label = pickle.load(open('label_list/' + sel, 'rb'))
            return render_template('predict.html', title=title, file=file_info_list, label=label, sel=sel)
            
        if proses == "2":
            sel = session["sel"]

            label = pickle.load(open('label_list/' + sel, 'rb'))
            models = pickle.load(open('model_training/' + sel, 'rb'))

            float_feature = [int(x) for x in request.form.values()]
            feature = np.delete(float_feature, -1)
            feature = feature.tolist()
            feature = [np.array(feature)]
            
            hslpre = []
            for itm in models:
                pre = itm[1].predict(feature)
                if pre == 1:
                    f = "= Positif Covid"
                else:
                    f = "= Tidak Covid"
                hslpre.append([itm[0], f])

            return render_template('predict.html', title=title, file=file_info_list, label=label, sel=sel, hslpre=hslpre, feature=feature, pr=models)


    return render_template('predict.html', title=title, file=file_info_list)

@app.route('/file', methods=['GET', 'POST'])
def file():
    title = "File"
    
    file = scan_files(path_input,".csv")

    if request.method == 'POST':
        itd = request.form['itd']
        itd = itd[:-4] if itd.endswith('.csv') else itd

        file_path = path_input + itd + ".csv"
        label = path_label + itd + ".pkl"
        model = path_model + itd + ".pkl"
        result = path_result + itd + ".pkl"

        if os.path.exists(file_path):
            os.remove(file_path)
            os.remove(label)
            os.remove(model)
            os.remove(result)
            file = scan_files(path_input,".csv")
            return render_template('file.html', title=title, file=file, alert=(f"Data " + itd + " Berhasil Di hapus"))
        else:
            return render_template('file.html', title=title, file=file, alert=(f"Data " + itd + " Tidak Ditemukan"))

    return render_template('file.html', title=title, file=file)



if __name__ == '__main__':
    app.run(port=3075, debug=True)