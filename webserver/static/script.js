let colorPicker;
let defaultColor = "#002300";
var socket = io();

let typeOfShutdown = ""; //reboot or Shutdown
//const defaultColor = "{{default_color}}";

window.addEventListener("load", startup, false);
function startup() {
colorPicker = document.querySelector("#color-picker");
defaultColor = document.getElementById("mydiv-color").dataset.color
colorPicker.value = defaultColor;
colorPicker.addEventListener("input", updateFirst, false);
colorPicker.addEventListener("change", updateAll, false);
colorPicker.select();
const p = document.getElementById("coloritem")
if (p) {
    p.style.background = defaultColor;
}
socket.emit('getvalues', {data: 'I\'m connected!'});
setInterval(OnButtonClickGetValues, 20500)
}
window.onclick = function(event) {
    var modal = document.getElementById("cardwrapper");

  if (event.target == modal) {
    modal.style.display = "none";
  }
}

function updateFirst(event) {
    //const p = document.querySelector("body");
    // if (p) {
    //     p.style.background = event.target.value;
    // }
}

function updateAll(event) {
    //document.querySelectorAll("p").forEach((p) => {
    //    p.style.color = event.target.value;
    //});
    const p = document.getElementById("coloritem")
    if (p) {
        p.style.background = defaultColor;
    }
}

socket.on('values', function(msg) {
    document.getElementById('total_images').innerHTML = msg.total_images;
    document.getElementById('total_collages').innerHTML = msg.total_collages;
    //document.getElementById('values').value = msg.color;

  });

  
socket.on('disk', function(msg) {
    document.getElementById('disk_free').innerHTML = msg.disk_free + " GB";
    //document.getElementById('disk_total').innerHTML = msg.disk_total + " GB";
    document.getElementById('disk_percentage').innerHTML = msg.disk_percentage + " %";
    document.getElementById('disk_percentage_bar').style.width = msg.disk_percentage + "%";

    
  });
  socket.on('cpu', function(msg) {
    document.getElementById('cpu_temp').innerHTML = msg.cpu_temp + " °C";
    //document.getElementById('disk_total').innerHTML = msg.disk_total + " GB";
    document.getElementById('cpu_percentage').innerHTML = msg.cpu_percentage + " %";
    document.getElementById('cpu_percentage_bar').style.width = msg.cpu_percentage + "%";

    
  });
  socket.on('time', function(msg) {
    document.getElementById('time').innerHTML = msg.time_now
  });
  socket.on('zipfiles', function(msg) {
    document.getElementById('zipfiles').innerHTML = msg.processed + " / " + msg.total;
  });


function OnButtonClickGetValues(){
    socket.emit('getvalues', {data: 'I\'m connected!'});
}

function alertBeforeShutdownReboot(location){

    document.getElementById('cardwrapper').style.display = "flex";

    typeOfShutdown = location
    if (location == 'reboot'){
        text = "neustarten";
        document.getElementById('cardbuttontext').textContent = "Neustart";
        document.getElementById('cardheading').innerHTML = "Fotobox neustarten?";
        document.getElementById('carddescription').innerHTML = "Die Fotobox startet sich neu. Alle Bilder bleiben erhalten..."

    }
    if (location == 'shutdown'){
        text = "herunterfahren";
        document.getElementById('cardbuttontext').textContent = "Herunterfahren";
        document.getElementById('cardheading').innerHTML = "Fotobox herunterfahren?";
        document.getElementById('carddescription').innerHTML = "Die Fotobox schaltet sich aus. Zum erneuten Starten der Fotobox den Strom aus- und einschalten."

    }
    //confirmtext = "Wollen Sie wirklich die Fotobox " + text + "?";
    //if (confirm(confirmtext) == true) {
       // socket.emit(location);
        //window.location.href = '/' + location;

    //}
}

function sendConfirmedShutdownReboot(){
   if(typeOfShutdown != ""){
    socket.emit(typeOfShutdown);
   }
   abort();
}

function abort(){
    document.getElementById('cardwrapper').style.display = "none";
}
