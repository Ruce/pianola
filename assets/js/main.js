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
{% include_relative midi.js %}


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
	initialiseComposeMode(piano.mode);
	initialiseTempoSettings(piano.defaultBpm, piano.autoDetectTempo);
	initialiseInputSettings(piano.autoCorrectInput);
	initialiseSeeds();
	initialiseRewindReceiver();
	initialiseSustainPedal();
	
	const historyPromise = loadHistory();
	const tonePromise = Tone.ToneAudioBuffer.loaded()
	
	Promise.all([historyPromise, tonePromise]).then(() => {
		NProgress.set(0.8);
		setTimeout(() => loadingComplete(), 300);
	});
	
	document.addEventListener('keydown', (event) => piano.keyDown(event));
	document.addEventListener('keyup', (event) => piano.keyUp(event));
	document.getElementById('introduction').onclick = function(event) { event.stopPropagation(); }
	document.getElementById('settingsMenu').onclick = function(event) { event.stopPropagation(); }
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

function initialiseComposeMode(initialMode) {
	const intialModeRadio = document.querySelector(`input[name="modeOptions"][value="${initialMode.name}"]`);
	if (intialModeRadio) intialModeRadio.checked;
	
	document.getElementById('modeSelectionForm').addEventListener('change', (event) => {
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
	});
}

function initialiseTempoSettings(initialBpm, initialAutoDetectTempo) {
	const slider = document.getElementById('tempoSlider');
	const numInput = document.getElementById('tempoInput');
	
	slider.value = initialBpm;
	numInput.value = initialBpm;
	
	slider.addEventListener('input', (event) => {
		const newBpm = Math.round(slider.value);
		numInput.value = newBpm;
		piano.pianoAudio.setBpm(newBpm);
	});
	
	numInput.addEventListener('change', (event) => {
		let newBpm = parseInt(numInput.value);
		const maxBpm = parseInt(numInput.max);
		const minBpm = parseInt(numInput.min);
		
		if (newBpm > maxBpm) {
			numInput.value = maxBpm;
			newBpm = maxBpm;
		} else if (newBpm < minBpm) {
			numInput.value = minBpm;
			newBpm = minBpm;
		}
		slider.value = newBpm;
		piano.pianoAudio.setBpm(newBpm);
	});
	
	numInput.addEventListener('keydown', (event) => event.stopPropagation());
	
	const autoDetectCheckbox = document.getElementById('tempoDetectEnable');
	autoDetectCheckbox.checked = initialAutoDetectTempo;
	toggleTempoInputs(autoDetectCheckbox.checked);
	
	autoDetectCheckbox.addEventListener("change", () => {
		piano.autoDetectTempo = autoDetectCheckbox.checked;
		toggleTempoInputs(autoDetectCheckbox.checked);
	});
}

function toggleTempoInputs(toDisable) {
	const slider = document.getElementById('tempoSlider');
	const numInput = document.getElementById('tempoInput');
	
	slider.disabled = toDisable;
	numInput.disabled = toDisable;
}

function initialiseInputSettings(initialAutoCorrectInput) {
	const autoCorrectCheckbox = document.getElementById('autoCorrectEnable');
	autoCorrectCheckbox.checked = initialAutoCorrectInput;
	
	autoCorrectCheckbox.addEventListener("change", () => {
		piano.autoCorrectInput = autoCorrectCheckbox.checked;
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

function initialiseSustainPedal() {
	const pedalMark = document.getElementById('sustainPedalMark');
	pedalMark.addEventListener('mousedown', () => piano.pressSustain());
	pedalMark.addEventListener('mouseup', () => piano.releaseSustain());
	pedalMark.addEventListener('mouseout', () => piano.releaseSustain());
	pedalMark.addEventListener('touchstart', (event) => piano.touchChangeSustain(event));
	pedalMark.addEventListener('touchend', (event) => piano.touchChangeSustain(event));
	pedalMark.addEventListener('touchcancel', (event) => piano.touchChangeSustain(event));
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

async function closeIntro() {
	// Ignore clicks on overlay if shared history is waiting to be played
	if (piano.sharedHistory) return;

	// As an entry point into the app, trigger AudioContext start
	await piano.pianoAudio.startTone();

	document.getElementById('introOverlay').style.display = 'none';
}

async function playShared() {
	// As an entry point into the app, trigger AudioContext start
	await piano.pianoAudio.startTone();
	
	piano.playSharedHistory();
	closeIntro();
}

function openSettings() {
	document.getElementById('settingsOverlay').style.display = 'block';
}

function closeSettings() {
	document.getElementById('settingsOverlay').style.display = 'none';
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