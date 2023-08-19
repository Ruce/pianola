---
---
{% include_relative model.js %}
{% include_relative piano.js %}
{% include_relative notes.js %}


const octaves = 5;

var piano;
var notesCanvas;
var globalMouseDown = false;
var endpointBaseUrl = 'https://vcnf5f6zo2.execute-api.eu-west-2.amazonaws.com/beta/next-notes?';

function initialisePage() {
	model = new PianolaModel(endpointBaseUrl);
	piano = new Piano('pianoCanvas', octaves, model);
	notesCanvas = new NotesCanvas('notesCanvas', piano);
}

function redrawCanvases() {
	piano.drawKeyboard();
	notesCanvas.draw();
}

function stopMusic() {
	if (typeof piano !== 'undefind') {
		piano.stopCallModel();
	}
}

document.addEventListener("DOMContentLoaded", initialisePage);
document.addEventListener("mouseup", () => globalMouseDown = false);

var resizeTimeout = false;
const resizeDelay = 40;
window.onresize = function () {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(redrawCanvases, resizeDelay);
}