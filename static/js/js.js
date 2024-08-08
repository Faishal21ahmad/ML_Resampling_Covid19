function openDropdown(idbtn, idlist) {
    const list = document.getElementById(idlist);
    list.classList.remove('hidden');
}

function closeDropDown(idbtn, idlist, namelist, idtxtbtn) {
    const list = document.getElementById(idlist);
    list.classList.add('hidden');

    const txtbtn = document.getElementById(idtxtbtn);
    txtbtn.innerText = namelist;
}

document.addEventListener('DOMContentLoaded', function () {
    setTimeout(function () {
        // Menambahkan kelas 'hidden' setelah 3 detik
        document.getElementById('alert').classList.add('hidden');
    }, 3000);
});




document.getElementById('uploadFi').addEventListener('click', function(event) {
    var fileInput = document.getElementById('file');
    var titleListInfo = document.getElementById('titlelistinfo');
    var listInfo = document.getElementById('listinfo');
    var msger = document.getElementById('msger');
    

    if (fileInput.files.length === 0) {
        // Hapus pesan error yang sudah ada
        titleListInfo.innerHTML = '';
        listInfo.innerHTML = '';
        // Tambahkan pesan error
        var errorParaTitle = document.createElement('p');
        errorParaTitle.className = 'p-1 text-red-500 font-semibold';
        errorParaTitle.innerText = 'Error';
        titleListInfo.appendChild(errorParaTitle);

        var errorParaList = document.createElement('p');
        errorParaList.className = 'p-1 my-auto text-red-500 font-semibold';
        errorParaList.innerText = '= File Tidak Boleh Kosong';
        listInfo.appendChild(errorParaList);
        document.getElementById('msger').classList.remove('hidden');

        event.preventDefault();
        return; // Hentikan eksekusi kode jika tidak ada file yang dipilih
    }
    document.getElementById('loading').classList.remove('hidden');
});


document.getElementById('runEvaluasi').addEventListener('click', function(event) {
    var selectedRadio = document.querySelector('input[name="kls"]:checked');
    var titleListInfo = document.getElementById('titlelistinfo');
    var listInfo = document.getElementById('listinfo');
    var msger = document.getElementById('msger');
    
    if (!selectedRadio) {
        // Hapus pesan error yang sudah ada
        titleListInfo.innerHTML = '';
        listInfo.innerHTML = '';
        // Tambahkan pesan error
        var errorParaTitle = document.createElement('p');
        errorParaTitle.className = 'p-1 text-red-500 font-semibold';
        errorParaTitle.innerText = 'Error';
        titleListInfo.appendChild(errorParaTitle);

        var errorParaList = document.createElement('p');
        errorParaList.className = 'p-1 my-auto text-red-500 font-semibold';
        errorParaList.innerText = '= Klasifikasi Tidak Boleh Kosong';
        listInfo.appendChild(errorParaList);
        document.getElementById('msger').classList.remove('hidden');

        event.preventDefault(); // Hentikan eksekusi kode jika tidak ada kolom yang dipilih
        return;
    }
    document.getElementById('loading').classList.remove('hidden');
});


// document.getElementById('prerun').addEventListener('click', function(event) {
//     document.getElementById('loading').classList.remove('hidden');
// });