---
---
{% include_relative actor.js %}
{% include_relative model.js %}
{% include_relative piano.js %}
{% include_relative notes.js %}


const octaves = 5;
const endpointBaseUrl = 'https://vcnf5f6zo2.execute-api.eu-west-2.amazonaws.com/beta/next-notes?';

var piano;
var notesCanvas;
var exampleSongs;
var globalMouseDown = false;

function hideLoader() {
	document.getElementById('loaderCircle').classList.add('complete');
	document.getElementById('loaderText').classList.add('complete');
	document.getElementById('loaderCheckmark').classList.add('draw');
	setTimeout(() => document.getElementById("connectionLoader").style.display = "none", 1700);
}

function initialisePage() {
	model = new PianolaModel(endpointBaseUrl);
	model.connectToModel(hideLoader);
	piano = new Piano('pianoCanvas', octaves, model);
	notesCanvas = new NotesCanvas('notesCanvas', piano);
	initialiseVolumeSlider();
}

function redrawCanvases() {
	piano.pianoCanvas.triggerDraw();
	notesCanvas.draw();
}

function initialiseVolumeSlider() {
	const slider = document.getElementById('volumeSlider');
	piano.changeVolume(slider.value);
	slider.addEventListener("input", (event) => {
		piano.changeVolume(event.target.value);
	});
}

function stopMusic() {
	if (typeof piano !== 'undefined') {
		piano.stopCallModel();
	}
}

function fetchSongExamples() {
	fetch('assets/songs/examples.json')
	.then(response => response.json())
	.then(data => exampleSongs = data)
	.catch(error => {
		console.log('Error reading song examples JSON:', error);
	});
}

function playExample(exampleNum) {
	const song = exampleSongs.songs[exampleNum-1];
	piano.playExample(song.data, song.bpm);
}

document.addEventListener("DOMContentLoaded", initialisePage);
document.addEventListener("DOMContentLoaded", fetchSongExamples);
document.addEventListener("mouseup", () => globalMouseDown = false);

var resizeTimeout = false;
const resizeDelay = 40;
window.onresize = function () {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(redrawCanvases, resizeDelay);
}