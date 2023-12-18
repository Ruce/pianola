---
---
{% include_relative actor.js %}
{% include_relative model.js %}
{% include_relative piano.js %}
{% include_relative pianoAudio.js %}
{% include_relative pianoCanvas.js %}
{% include_relative pianoKey.js %}
{% include_relative pianoMode.js %}
{% include_relative pianoRoll.js %}
{% include_relative history.js %}
{% include_relative note.js %}
{% include_relative notesCanvas.js %}


const octaves = 5;
const ticksPerBeat = 8;
const endpointBaseUrl = 'https://vcnf5f6zo2.execute-api.eu-west-2.amazonaws.com/alpha/generate?';
const databaseBaseUrl = 'https://t4z5bpis3b.execute-api.eu-west-2.amazonaws.com/beta/PianolaHistoryDDB?'

var piano;
var notesCanvas;
var exampleSongs;
var globalMouseDown = false;

function initialisePage() {
	NProgress.configure({ showSpinner: false, trickle: true });
	NProgress.start();
	
	model = new PianolaModel(endpointBaseUrl);
	connectToModel(model);
	
	historyController = new HistoryController(databaseBaseUrl);
	piano = new Piano('pianoCanvas', octaves, ticksPerBeat, model, historyController);
	notesCanvas = new NotesCanvas('notesCanvas', piano);
	
	initialiseVolumeSlider();
	initialiseSeeds();
	initialiseRewindReceiver();
	closeMode(false); // Sets the mode radio button to the default in `piano`
	
	const historyPromise = loadHistory();
	const tonePromise = Tone.ToneAudioBuffer.loaded()
	
	Promise.all([historyPromise, tonePromise]).then(() => {
		NProgress.set(0.8);
		setTimeout(() => loadingComplete(), 300);
	});
	
	document.addEventListener('keydown', (event) => piano.keyDown(event));
	document.addEventListener('keyup', (event) => piano.keyUp(event));
	document.getElementById('introduction').onclick = function(event) { event.stopPropagation(); }
	document.getElementById('modeMenu').onclick = function(event) { event.stopPropagation(); }
}

function showLoader() {
	document.getElementById('loaderCircle').classList.remove('complete');
	document.getElementById('loaderText').classList.remove('complete');
	document.getElementById('loaderCheckmark').classList.remove('draw');
	document.getElementById('connectionLoader').style.display = 'block';
}

function hideLoader() {
	document.getElementById('loaderCircle').classList.add('complete');
	document.getElementById('loaderText').classList.add('complete');
	document.getElementById('loaderCheckmark').classList.add('draw');
	setTimeout(() => document.getElementById('connectionLoader').style.display = 'none', 1800);
}

function connectToModel(model) {
	showLoader();
	model.isConnected = false;
	model.connectToModel(hideLoader);
}

function loadingComplete() {
	document.getElementById('content').style.display = 'block';
	document.getElementById('preloader').classList.add('fade-out');
	
	redrawCanvases();
	NProgress.done();
}

function redrawCanvases() {
	const pianoDiv = document.getElementById('pianoDiv');
	const pianoHeight = window.innerWidth * PianoCanvas.keyboardRatio;
	pianoDiv.style.height = pianoHeight + 'px';
	piano.pianoCanvas.triggerDraw();
	notesCanvas.draw();
	for (const pianoRoll of piano.historyController.allPianoRolls) {
		pianoRoll.draw();
	}
}

var previousVolume = 0;
function initialiseVolumeSlider() {
	const slider = document.getElementById('volumeSlider');
	previousVolume = Math.round(slider.value);
	piano.pianoAudio.changeVolume(previousVolume);
	
	slider.addEventListener('input', (event) => {
		const newVolume = Math.round(event.target.value);
		previousVolume = newVolume;
		piano.pianoAudio.changeVolume(newVolume);
		
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

function initialiseSeeds() {
	fetch('assets/songs/examples.json')
	.then(response => response.json())
	.then(data => populateSeedList(data))
	.catch(error => {
		console.log('Error reading song examples JSON:', error);
	});
}

function initialiseRewindReceiver() {
	const rewindReceiver = document.getElementById('rewindReceiver');
	
	function rewindAnimated() {
		const rewindSuccess = piano.rewind();
		if (rewindSuccess) {
			rewindReceiver.style.opacity = 0.9;
			setTimeout(() => rewindReceiver.style.opacity = 0, 350);
		}
	}
	rewindReceiver.addEventListener('dblclick', () => rewindAnimated());
}

async function loadHistory() {
	const searchParams = new URLSearchParams(window.location.search);
	if (searchParams.get('play')) {
		await piano.loadSharedHistory(searchParams.get('play'));
	}
}

function populateSeedList(exampleSongs) {
	const numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV'];
	const listContainer = document.getElementById('exampleList');
	for (let i = 0; i < exampleSongs.songs.length; i++) {
		const seed = exampleSongs.songs[i];
		const seedElement = document.createElement('li');
		const seedTitle = `Seed ${numerals[i]}: ${seed.name}`;
		seedElement.textContent = seedTitle;
		seedElement.addEventListener('click', () => piano.playExample(seed.data, seed.bpm, seedTitle));
		listContainer.appendChild(seedElement);
	}
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
		piano.pianoAudio.changeVolume(newVolume);
		document.getElementById('menuMuteTooltip').textContent = 'Mute';
	} else {
		volumeButtonImage.classList.add('volumeButtonOff');
		volumeSlider.value = 0;
		piano.pianoAudio.changeVolume(0);
		document.getElementById('menuMuteTooltip').textContent = 'Unmute';
	}
}

function rewindMusic() {
	if (typeof piano !== 'undefined') {
		piano.rewind();
	}
}

function stopMusic() {
	if (typeof piano !== 'undefined') {
		piano.resetAll();
	}
}

function openIntro() {
    document.getElementById('introOverlay').style.display = 'block';
	document.getElementById('introText').style.display = 'block';
	document.getElementById('closeIntroButton').style.display = 'block';
	document.getElementById('introShared').style.display = 'none';
}

function closeIntro() {
    // Close the overlay if there isn't a shared history waiting to be played
	if (!piano.sharedHistory) {
		document.getElementById('introOverlay').style.display = 'none';
	}
}

function playShared() {
	piano.playSharedHistory();
	closeIntro();
}

function openMode() {
	document.getElementById('modeOverlay').style.display = 'block';
}

function closeMode(toSave) {
	document.getElementById('modeOverlay').style.display = 'none';
	
	if (toSave) {
		const selectedRadio = document.querySelector('input[name="modeOptions"]:checked');
		if (selectedRadio) {
			const newMode = PianoMode.getModeByName(selectedRadio.value);
			if (piano.mode !== newMode) {
				piano.mode = newMode;
				piano.resetAll();
				if (piano.mode === PianoMode.Composer || piano.mode === PianoMode.Autoplay) {
					connectToModel(model);
				}
			}
		}
	} else {
		const radioButtons = document.querySelectorAll('input[name="modeOptions"]');
		for (const radioButton of radioButtons) {
			if (radioButton.value === piano.mode.name) {
				radioButton.checked = true;
			}
		}
	}
}

function pauseFramesCheck() {
	if (notesCanvas && document.visibilityState === 'hidden') {
		notesCanvas.lastAnimationCheck = null;
	}
}

function toggleHistoryDrawer() {
	document.getElementById('historyDrawer').classList.toggle('open');
	document.getElementById('drawerToggleShape').classList.toggle('open');
}

function closeHistoryDrawer() {
	document.getElementById('historyDrawer').classList.remove('open');
	document.getElementById('drawerToggleShape').classList.remove('open');
}

function toggleHeartIcon(event) {
	event.stopPropagation();
	event.target.classList.toggle('fa-regular');
	event.target.classList.toggle('fa-solid');
}

function dismissShareLinkTooltip(event) {
	function getAncestors(element) {
		const ancestors = [];
		let parent = element.parentNode;
		
		while (parent) {
			ancestors.push(parent);
			parent = parent.parentNode;
		}
		return ancestors;
	}
	
	const containers = Array.from(document.getElementsByClassName('shareLinkContainer'));
	if (containers.length === 0) return;
	
	// Get the ancestor elements of the event target to check if it is a share link tooltip
	const targetAncestors = getAncestors(event.target);
	for (const container of containers) {
		if (event.target !== container && !targetAncestors.includes(container)) {
			container.remove();
		}
	}
}

document.addEventListener('DOMContentLoaded', initialisePage);
document.addEventListener('visibilitychange', pauseFramesCheck);
document.addEventListener('mouseup', () => globalMouseDown = false);
document.addEventListener('click', dismissShareLinkTooltip);

var resizeTimeout = false;
const resizeDelay = 40;
window.onresize = function () {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(redrawCanvases, resizeDelay);
}