let colorPicker;
let defaultColor = "#002300";
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



function alertBeforeShutdownReboot(location){

    if (location == 'reboot'){
        text = "neustarten";
    }
    if (location == 'shutdown'){
        text = "herunterfahren";
    }
    confirmtext = "Wollen Sie wirklich die Fotobox " + text + "?";
    if (confirm(confirmtext) == true) {
        window.location.href = '/' + location;
    }
}
