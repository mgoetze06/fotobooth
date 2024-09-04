let colorPicker;
let defaultColor = "#002300";

window.addEventListener("load", startup, false);

function startup() {
    defaultColor = document.getElementById("mydiv-color").dataset.color


    colorPicker = document.querySelector("#color-picker");
    //defaultColor = colorPicker.value;
    colorPicker.value = defaultColor;
    colorPicker.addEventListener("input", updateFirst, false);
    colorPicker.addEventListener("change", updateAll, false);
    colorPicker.select();
    const p = document.querySelector("body");
        if (p) {
            p.style.background = defaultColor;
        }
}
function updateFirst(event) {
    const p = document.querySelector("body");
    if (p) {
        p.style.background = event.target.value;
    }
}

function updateAll(event) {
    //document.querySelectorAll("p").forEach((p) => {
    //    p.style.color = event.target.value;
    //});
}
//if ( window.history.replaceState ) {
//    window.history.replaceState( null, null, window.location.href );
//}