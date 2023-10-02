---
---
{% include_relative actor.js %}
{% include_relative model.js %}
{% include_relative piano.js %}
{% include_relative notes.js %}


const octaves = 5;
//const endpointBaseUrl = 'https://vcnf5f6zo2.execute-api.eu-west-2.amazonaws.com/beta/next-notes?';
const endpointBaseUrl = 'https://vcnf5f6zo2.execute-api.eu-west-2.amazonaws.com/beta/generate?';

var piano;
var notesCanvas;
var exampleSongs;
var globalMouseDown = false;

function hideLoader() {
	document.getElementById('loaderCircle').classList.add('complete');
	document.getElementById('loaderText').classList.add('complete');
	document.getElementById('loaderCheckmark').classList.add('draw');
	setTimeout(() => document.getElementById("connectionLoader").style.display = "none", 1800);
}

function initialisePage() {
	NProgress.configure({ showSpinner: false });
	NProgress.start();
	
	model = new PianolaModel(endpointBaseUrl);
	model.connectToModel(hideLoader);
	piano = new Piano('pianoCanvas', octaves, model);
	notesCanvas = new NotesCanvas('notesCanvas', piano);
	initialiseVolumeSlider();
	fetchSongExamples();
	
	Tone.ToneAudioBuffer.loaded().then(() => {
		NProgress.set(0.8);
		setTimeout(() => loadingComplete(), 500);
	});
}

function loadingComplete() {
	document.getElementById('content').style.display = 'block';
	document.getElementById('preloader').classList.add('fade-out');
	NProgress.done()
}

function redrawCanvases() {
	piano.pianoCanvas.triggerDraw();
	notesCanvas.draw();
}

var previousVolume = 0;
function initialiseVolumeSlider() {
	const slider = document.getElementById('volumeSlider');
	previousVolume = Math.round(slider.value);
	piano.changeVolume(previousVolume);
	
	slider.addEventListener("input", (event) => {
		const newVolume = Math.round(event.target.value);
		previousVolume = newVolume;
		piano.changeVolume(newVolume);
		
		// Change the speaker icon to on or off if necessary
		const volumeButtonImage = document.getElementById('volumeButtonImage');
		if (newVolume === 0 && !volumeButtonImage.classList.contains('volumeButtonOff')) {
			volumeButtonImage.classList.add('volumeButtonOff');
			document.getElementById('menuMuteTooltip').textContent = 'Unmute';
		} else if (newVolume > 0 && volumeButtonImage.classList.contains('volumeButtonOff')) {
			volumeButtonImage.classList.remove('volumeButtonOff');
			document.getElementById('menuMuteTooltip').textContent = 'Mute';
		}
	});
}

function fetchSongExamples() {
	fetch('assets/songs/examples.json')
	.then(response => response.json())
	.then(data => exampleSongs = data)
	.catch(error => {
		console.log('Error reading song examples JSON:', error);
	});
}

function toggleMute() {
	const volumeSlider = document.getElementById('volumeSlider');
	const volumeButtonImage = document.getElementById('volumeButtonImage');
	const wasMuted = volumeButtonImage.classList.contains('volumeButtonOff');
	if (wasMuted) {
		volumeButtonImage.classList.remove('volumeButtonOff');
		const newVolume = (previousVolume === 0) ? 15 : previousVolume;
		previousVolume = newVolume;
		volumeSlider.value = newVolume;
		piano.changeVolume(newVolume);
		document.getElementById('menuMuteTooltip').textContent = 'Mute';
	} else {
		volumeButtonImage.classList.add('volumeButtonOff');
		volumeSlider.value = 0;
		piano.changeVolume(0);
		document.getElementById('menuMuteTooltip').textContent = 'Unmute';
	}
}

function stopMusic() {
	if (typeof piano !== 'undefined') {
		piano.stopCallModel();
	}
}

function playExample(exampleNum) {
	const song = exampleSongs.songs[exampleNum-1];
	piano.playExample(song.data, song.bpm);
}

function openOverlay() {
    document.getElementById('overlay').style.display = 'block';
}

function closeOverlay() {
    document.getElementById('overlay').style.display = 'none';
}

document.addEventListener("DOMContentLoaded", initialisePage);
document.addEventListener("mouseup", () => globalMouseDown = false);
document.addEventListener("keydown", (event) => piano.keyDown(event));
document.addEventListener("keyup", (event) => piano.keyUp(event));

var resizeTimeout = false;
const resizeDelay = 40;
window.onresize = function () {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(redrawCanvases, resizeDelay);
}