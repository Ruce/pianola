class Piano {
	constructor(canvasId, octaves, model) {
		if (octaves < 1 || octaves > 7) {
			throw new RangeError("The number of octaves must be between 1 and 7");
		}
		this.octaves = octaves;
		this.pianoKeys = Piano.createPianoKeys(this.octaves);
		this.keyMap = PianoKeyMap.keyMap;
		this.pianoCanvas = new PianoCanvas(this, canvasId);
		this.sampler = this.initialiseSampler();
		this.model = model;
		
		this.toneStarted = false;
		this.lastActivity = new Date();
		this.modelStartTime = new Date();
		this.activeNotes = [];
		this.noteHistory = [];
		this.noteQueue = [];
		
		this.bufferBeats = 4;
		this.bufferTicks = Tone.Time(`0:${this.bufferBeats}`).toTicks();
		this.historyWindowBeats = 60;
		this.defaultBPM = 80;
		this.setBPM(this.defaultBPM);
	}
	
	static createPianoKeys(octaves) {
		const numKeys = (7 * octaves + 3) + (5 * octaves + 1); // White keys + black keys, and extra keys outside of main octaves
		const lowestMidiNote = PianoKey.midiNoteNumMiddleC - (Math.floor(octaves / 2) * 12) - 3; // Calculate lowest note from middle C
		const pianoKeys = [];
		for (let i = 0; i < numKeys; i++) {
			pianoKeys.push(new PianoKey(i, lowestMidiNote + i));
		}
		return pianoKeys;
	}
	
	static circVar(samples, high=1, low=0) {
		let sinTotal = 0;
		let cosTotal = 0;
		for (const n of samples) {
			sinTotal += Math.sin(((n - low) * 2 * Math.PI) / (high - low));
			cosTotal += Math.cos(((n - low) * 2 * Math.PI) / (high - low));
		}
		const sinMean = sinTotal / samples.length;
		const cosMean = cosTotal / samples.length;
		const hypot = Math.sqrt(Math.pow(sinMean, 2) + Math.pow(cosMean, 2));
		
		return Math.min(1 - hypot, 1);
	}

	static calcDispersion(noteTimings, interval) {
		const deltas = Array.from(noteTimings, (t) => (t % interval) / interval);
		return Piano.circVar(deltas, 1, 0);
	}

	static calcBestInterval(noteTimings, low, high, bias=0) {
		// `bias`: add a bias to the dispersion of lower intervals to prefer high intervals
		let lowestDispersion = null;
		let bestInterval = -1;
		
		for (let i = low; i <= high; i++) {
			let dispersion = Piano.calcDispersion(noteTimings, i) + (bias * (high - i));
			if (lowestDispersion === null || dispersion < lowestDispersion) {
				lowestDispersion = dispersion;
				bestInterval = i;
			}
		}
		return bestInterval;
	}
	
	beatsToSeconds(beats) {
		return (beats / this.bpm) * 60;
	}
	
	detectBpm(lowestBpm=60, highestBpm=110) {
		const ticksPerSec = 480;
		const noteTimings = Array.from(this.noteHistory, (n) => n.time * ticksPerSec);
		const lowInterval = Math.floor(ticksPerSec * (60 / highestBpm) / 4);
		const highInterval = Math.ceil(ticksPerSec * (60 / lowestBpm) / 4);
		
		const bestInterval = Piano.calcBestInterval(noteTimings, lowInterval, highInterval, 0.002);
		const bestBpm = Math.round(ticksPerSec * 60 / (bestInterval * 4));
		console.log('Detected bpm:', bestBpm);
		return bestBpm;
	}
	
	keyPressed(keyNum) {
		if (this.startTone()) {
			this.seedInputListener();
		}
		this.playNote(this.pianoKeys[keyNum], 0.8, -1, Actor.Player);
		this.lastActivity = new Date();
	}
	
	keyDown(event) {
		if (event.repeat) return;
		if (event.key === ' ') {
			this.stopModel();
		}
		
		const keyNum = this.keyMap[event.key];
		if (keyNum !== undefined) {
			this.keyPressed(keyNum);
		}
	}
	
	keyUp(event) {
		this.lastActivity = new Date();
		const keyNum = this.keyMap[event.key];
		if (keyNum !== undefined) {
			this.releaseNote(this.pianoKeys[keyNum]);
		}
	}
	
	playNote(pianoKey, velocity, duration, actor, contextTime, triggerTime=Tone.Transport.seconds) {
		// Check if pianoKey is already active, i.e. note is played again while currently held down, and if so release it
		this.releaseNote(pianoKey, triggerTime);
		if (duration === -1) {
			// Note is being held down
			this.sampler.triggerAttack(pianoKey.keyName, contextTime, velocity);
		} else {
			const triggerDuration = duration + 0.15; // Add a short delay to the end of the sound (in seconds)
			this.sampler.triggerAttackRelease(pianoKey.keyName, triggerDuration, contextTime, velocity);
		}
		
		// Keep track of note in `activeNotes` and `noteHistory`
		const note = new Note(pianoKey, velocity, duration, triggerTime, actor);
		this.activeNotes.push(note);
		this.noteHistory.push(note);
		
		// Draw note on canvases
		const currTime = new Date();
		let endTime = -1;
		if (duration !== -1) {
			endTime = new Date(currTime);
			const durationMilliseconds = (duration * 1000) * 0.9 - 15; // Shorten duration slightly to add a gap between notes
			endTime.setMilliseconds(endTime.getMilliseconds() + durationMilliseconds);
		}
		this.pianoCanvas.triggerDraw();
		this.notesCanvas.addNoteBar(pianoKey, currTime, endTime, actor);
		
		// Remove note from noteQueue if exists
		const queuedNote = this.noteQueue.find(note => note.key === pianoKey && note.actor === actor && note.time === triggerTime);
		const noteIndex = this.noteQueue.indexOf(queuedNote);
		if (noteIndex > -1) this.noteQueue.splice(noteIndex, 1);
	}
	
	releaseNote(pianoKey, triggerTime=Tone.Transport.seconds) {
		// Check if `pianoKey` is in the `activeNotes` array, and if so release the note
		const activeNote = this.activeNotes.find(note => note.key === pianoKey);
		if (activeNote !== undefined) {
			this.sampler.triggerRelease(pianoKey.keyName, Tone.now() + 0.1);
			this.activeNotes.splice(this.activeNotes.indexOf(activeNote), 1);
			this.notesCanvas.releaseNote(pianoKey.keyNum, new Date());
			
			if (activeNote.duration === -1) {
				// Update activeNote's final duration, which is also recorded in `noteHistory`
				activeNote.duration = triggerTime - activeNote.time;
			}
		}
	}
	
	releaseAllNotes() {
		for (const note of this.activeNotes) {
			this.releaseNote(note.key);
		}
	}
	
	scheduleNote(pianoKey, velocity, duration, triggerTime, actor) {
		const scheduleId = Tone.Transport.scheduleOnce((time) => this.playNote(pianoKey, velocity, duration, actor, time, triggerTime), triggerTime);
		this.noteQueue.push(new Note(pianoKey, velocity, duration, triggerTime, actor, scheduleId));
	}
	
	async callModel() {
		const initiatedTime = new Date();
		const offset = this.beatsToSeconds(1/4) / 2; // Start the window at half of an interval (sixteenth note) earlier so that notes are centered
		const start = Math.max(this.callModelEnd - this.beatsToSeconds(this.historyWindowBeats), 0) - offset;
		const history = Note.getRecentHistory(this.noteHistory, start);
		const queued = Note.getRecentHistory(this.noteQueue, start);
		history.push(...queued);
		
		const generated = await this.model.generateNotes(history, start, this.callModelEnd, this.bpm, this.bufferBeats * 4);
		// Before scheduling notes, check that the model hasn't been restarted while this function was awaiting a response
		if (initiatedTime >= this.modelStartTime) {
			for (const note of generated) {
				this.scheduleNote(this.pianoKeys[note.keyNum], note.velocity, note.duration, note.time, Actor.Model);
			}
			this.callModelEnd += this.beatsToSeconds(this.bufferBeats);
		}
	}
	
	checkActivity() {
		const timeout = 5 * 60 * 1000; // in milliseconds
		if (new Date() - this.lastActivity > timeout) {
			console.log('No user activity, stopping model...');
			this.stopModel();
		}
	}
	
	startModel() {
		this.modelStartTime = new Date();
		this.callModel(); // Call model immediately, since setInterval first triggers function after the delay
		this.callModelIntervalId = setInterval(() => this.callModel(), this.beatsToSeconds(this.bufferBeats) * 1000);
		this.checkActivityIntervalId = setInterval(() => this.checkActivity(), 5000);
		this.notesCanvas.startGlow(this.beatsToSeconds(this.bufferBeats) * 1000);
	}
	
	stopModel() {
		this.releaseAllNotes();
		this.notesCanvas.endGlow();
		Tone.Transport.cancel();
		Tone.Transport.stop();
		
		this.noteHistory = [];
		this.noteQueue = [];
		this.model.noteHistory = [];
		this.modelStartTime = new Date();
		this.toneStarted = false;
		
		if (typeof this.callModelIntervalId !== 'undefined') {
			clearInterval(this.callModelIntervalId);
		}
		if (typeof this.checkActivityIntervalId !== 'undefined') {
			clearInterval(this.checkActivityIntervalId);
		}
		if (typeof this.listenerIntervalId !== 'undefined') {
			clearInterval(this.listenerIntervalId);
			document.getElementById("listener").style.visibility = "hidden";
		}
	}
	
	seedInputListener() {
		this.setBPM(this.defaultBPM);
		
		// Display the listening visual indicator
		document.getElementById("listener").style.visibility = "visible";
		
		// Start up the awaiter
		this.listenerIntervalId = setInterval(this.seedInputAwaiter.bind(this), 100);
	}
	
	seedInputAwaiter() {
		const currTime = new Date();
		if (currTime - this.lastActivity > 2500 && this.activeNotes.length === 0) {
			clearInterval(this.listenerIntervalId);
			
			// Detect tempo from user input
			this.bpm = this.detectBpm();
			
			// Add an interval (sixteenth note) at the end of user passage to smooth out playback
			this.callModelEnd = this.noteHistory.at(-1).time + this.beatsToSeconds(1/4);
			
			// Rewind the transport schedule to the last of the seed input so that the history fed to the model is seamless
			Tone.Transport.seconds = this.callModelEnd - this.beatsToSeconds(this.bufferBeats);
			this.startModel();
			
			// Hide the listening visual indicator
			document.getElementById("listener").style.visibility = "hidden";
		}
	}
	
	playExample(data, bpm) {
		this.stopModel();
		this.setBPM(bpm);
		this.startTone();
		this.lastActivity = new Date();
		
		const start = Tone.Transport.seconds;
		const notes = PianolaModel.queryStringToNotes(data, start, this.bpm);
		for (const note of notes) {
			this.scheduleNote(this.pianoKeys[note.keyNum], note.velocity, note.duration, note.time, Actor.Bot);
		}
		
		// Schedule startModel to trigger one buffer period before the last note
		this.callModelEnd = notes.at(-1).time;
		Tone.Transport.scheduleOnce(() => this.startModel(), Math.max(0, this.callModelEnd - this.beatsToSeconds(this.bufferBeats)));
	}
	
	changeVolume(volume) {
		const volumeDb = (volume < 1) ? -Infinity : -(40 - (volume/3));
		this.sampler.volume.value = volumeDb;
	}
	
	initialiseSampler() {
		const noteKeys = this.pianoKeys.map((k) => k.keyName); // Get a list of all notes e.g. ['A3', 'A#3', 'B3', 'C4'...]
		const sampleFiles = Object.assign({}, ...noteKeys.map((n) => ({[n]: n.replace('#', 's') + ".mp3"})));
		// No sample files for keys A0, A#0, and B0
		delete sampleFiles['A0'];
		delete sampleFiles['A#0']
		delete sampleFiles['B0']
		
		const sampler = new Tone.Sampler({
			urls: sampleFiles,
			baseUrl: "assets/samples/piano/",
			release: 0.3,
			volume: -8
		}).toDestination();
		
		return sampler;
	}
	
	setBPM(bpm) {
		Tone.Transport.bpm.value = bpm;
		this.bpm = bpm;
		if (typeof this.notesCanvas !== 'undefined') {
			this.notesCanvas.setBPM(bpm);
		}
	}
	
	startTone() {
		// Returns true if Tone was just started, otherwise returns false (i.e. if Tone had already been started)
		if (!this.toneStarted) { 
			Tone.start().then(() => {
				Tone.Transport.start();
			});
			this.toneStarted = true;
			return true;
		} else {
			return false;
		}
	}
	
	bindNotesCanvas(notesCanvas) {
		this.notesCanvas = notesCanvas;
		this.notesCanvas.setBPM(this.defaultBPM);
	}
}

class PianoCanvas {
	constructor(piano, canvasId) {
		this.piano = piano;
		this.octaves = piano.octaves;
		this.numWhiteKeys = 7 * this.octaves + 3; // 3 additional keys before and after main octaves
		this.numBlackKeys = 5 * this.octaves + 1; // 1 additional key in the 0th octave
		this.whiteHotkeyMap = this.getHotkeyMap(true);
		this.blackHotkeyMap = this.getHotkeyMap(false);
		
		this.hoverKey = null;
		this.prevHoverKey = null;
		this.touchedKeys = [];
		this.animationQueued = false;
		
		this.canvas = document.getElementById(canvasId);
		this.canvas.style.touchAction = 'none';
		this.canvas.addEventListener('mousedown', this.mouseDownKeyboard.bind(this));
		this.canvas.addEventListener('mousemove', this.mouseMoveKeyboard.bind(this));
		this.canvas.addEventListener('mouseup', this.mouseUpKeyboard.bind(this));
		this.canvas.addEventListener('mouseout', this.mouseOutKeyboard.bind(this));
		this.canvas.addEventListener('touchstart', this.touchChangeKeyboard.bind(this));
		this.canvas.addEventListener('touchmove', this.touchChangeKeyboard.bind(this));
		this.canvas.addEventListener('touchend', this.touchChangeKeyboard.bind(this));
		this.canvas.addEventListener('touchcancel', this.touchChangeKeyboard.bind(this));
		this.triggerDraw();
	}
	
	static get keyboardRatio() {
		return 1/8;
	}
	
	static get blackKeyWidthRatio() {
		return 1/2;
	}
	
	static get blackKeyHeightRatio() {
		return 2/3;
	}
	
	// Top left coordinate of each black key relative to start of an octave (normalised by whiteKeyWidth)
	static get blackKeyPos() {
		return [
			2/3,
			1 + 5/6,
			3 + 5/8,
			4 + 3/4,
			5 + 7/8
		];
	}
	
	static get keyFill() {
		return {
			white: {inactive: '#FEFEFE', hover: '#FFF9D7', player: '#FEF3B0', model: '#D2EEEF', bot: '#CEFEDF'},
			black: {inactive: '#595959', hover: '#D0AD40', player: '#C09200', model: '#478C8F', bot: '#2C6E4E'}
		};
	}
	
	getKeyNumByCoord(clientX, clientY) {
		const canvasRect = this.canvas.getBoundingClientRect();
		const x = clientX - canvasRect.left;
		const y = clientY - canvasRect.top;
		
		const octaveWidth = this.whiteKeyWidth * 7;
		const o = Math.floor((x + (this.whiteKeyWidth * 5)) / octaveWidth); // Current octave
		const deltaX = x - ((o-1) * octaveWidth) - (2 * this.whiteKeyWidth); // x position relative to octave
		
		if (y > this.blackKeyHeight) {
			// Must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = PianoKey.whiteKeyNumbers[n];
			const keyNum = (octaveKeyNum - 1) + o*12 - 9;
			return keyNum;
		} else if (o === this.octaves + 1) {
			// Only highest C is in the highest octave
			const keyNum = o * 12 - 9;
			return keyNum;
		} else {
			for (let i=0; i < PianoCanvas.blackKeyPos.length; i++) {
				if (o === 0 && i < 4) {
					// 0th octave does not have first 4 black keys
					continue;
				}
				const pos = PianoCanvas.blackKeyPos[i];
				const blackKeyLeft = this.whiteKeyWidth * pos;
				const blackKeyRight = blackKeyLeft + this.blackKeyWidth;
				// Except for octave 0, which only has 1 black key
				if (deltaX >= blackKeyLeft && deltaX <= blackKeyRight) {
					const octaveKeyNum = PianoKey.blackKeyNumbers[i];
					const keyNum = (octaveKeyNum - 1) + o*12 - 9;
					return keyNum;
				}
			}
			// Not a black key, therefore must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = PianoKey.whiteKeyNumbers[n];
			const keyNum = (octaveKeyNum - 1) + o*12 - 9;
			return keyNum;
		}
	}
	
	getXCoordByKey(isWhiteKey, colourKeyNum) {
		if (isWhiteKey) {
			return this.whiteKeyWidth * colourKeyNum;
		} else {
			const k = (colourKeyNum + 4) % 5; // Index of the 5 black keys in `blackKeyPos`
			const o = Math.floor((colourKeyNum-1) / 5); // Current octave (first full octave is index 0, unlike PianoKey convention)
			return this.whiteKeyWidth * (PianoCanvas.blackKeyPos[k] + o*7 + 2);
		}
	}
	
	getHotkeyMap(isWhite) {
		// Calculate the hotkey for each colourKeyNum
		const hotkeyMap = {};
		const numKeys = isWhite ? this.numWhiteKeys : this.numBlackKeys;
		for (let i = 0; i < numKeys; i++) {
			const keyNum = PianoKey.calcKeyNumFromColourKeyNum(i, isWhite);
			const hotkey = Object.keys(this.piano.keyMap).find(key => this.piano.keyMap[key] === keyNum);
			if (hotkey !== undefined) {
				hotkeyMap[i] = hotkey.toUpperCase();
			}
		}
		return hotkeyMap;
	}
	
	mouseDownKeyboard(event) {
		globalMouseDown = true;
		const clickedKeyNum = this.getKeyNumByCoord(event.clientX, event.clientY);
		this.piano.keyPressed(clickedKeyNum);
	}
	
	mouseMoveKeyboard(event) {
		const hoverKeyNum = this.getKeyNumByCoord(event.clientX, event.clientY);
		this.hoverKey = this.piano.pianoKeys[hoverKeyNum];
		if (this.prevHoverKey === null || this.prevHoverKey.keyNum !== this.hoverKey.keyNum) {
			// Newly moused over key
			this.triggerDraw();
			if (globalMouseDown) {
				this.piano.releaseNote(this.prevHoverKey);
				this.piano.keyPressed(hoverKeyNum);
			}
			this.prevHoverKey = this.hoverKey;
		}
	}
	
	mouseUpKeyboard(event) {
		this.piano.releaseNote(this.hoverKey);
		this.triggerDraw();
		this.piano.lastActivity = new Date();
	}
	
	mouseOutKeyboard(event) {
		if (globalMouseDown && this.hoverKey !== null) this.piano.releaseNote(this.hoverKey);
		this.hoverKey = null;
		this.prevHoverKey = null;
		this.triggerDraw();
		this.piano.lastActivity = new Date();
	}
	
	touchChangeKeyboard(event) {
		event.preventDefault();
		const currTouchedKeys = [];
		for (const t of event.targetTouches) {
			const touchedElement = document.elementFromPoint(t.clientX, t.clientY); // Touch could have moved outside of origin element
			if (touchedElement && touchedElement.id === 'pianoCanvas') {
				const keyNum = this.getKeyNumByCoord(t.clientX, t.clientY);
				currTouchedKeys.push(this.piano.pianoKeys[keyNum]);
			}
		}
		
		const newKeys = currTouchedKeys.filter(x => !this.touchedKeys.includes(x));
		const releasedKeys = this.touchedKeys.filter(x => !currTouchedKeys.includes(x));
		
		for (const k of newKeys) {
			this.piano.keyPressed(k.keyNum);
			this.touchedKeys.push(k);
		}
		
		for (const k of releasedKeys) {
			this.piano.releaseNote(k);
			this.touchedKeys.splice(this.touchedKeys.indexOf(k), 1); // Remove key
		}
		
		if (newKeys.length + releasedKeys.length > 0) this.triggerDraw();
	}
	
	triggerDraw() {
		if (!this.animationQueued) {
			this.animationQueued = true;
			window.requestAnimationFrame(() => this.drawKeyboard());
		}
	}
	
	drawKeyboard() {
		const hoverKeyDefined = (this.hoverKey !== null);
		
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = window.innerWidth;
		this.canvas.height = this.canvas.width * PianoCanvas.keyboardRatio;
		ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

		this.whiteKeyWidth = this.canvas.width / this.numWhiteKeys;
		this.whiteKeyHeight = this.canvas.height;
		this.blackKeyWidth = this.whiteKeyWidth * PianoCanvas.blackKeyWidthRatio;
		this.blackKeyHeight = this.whiteKeyHeight * PianoCanvas.blackKeyHeightRatio;
		const [whiteKeyWidth, whiteKeyHeight, blackKeyWidth, blackKeyHeight] = [this.whiteKeyWidth, this.whiteKeyHeight, this.blackKeyWidth, this.blackKeyHeight];
		
		const keyFont = this.canvas.width > 800 ? "13px sans-serif" : "8px sans-serif";
		const biggerKeyFont = "24px sans-serif"; // For comma (,) and period (.)
		ctx.font = keyFont;
		
		// Remove expired keys and get the colourKeyNums for active keys
		this.piano.activeNotes = this.piano.activeNotes.filter(n => n.duration === -1 || n.time + n.duration < Tone.Transport.seconds);
		function getActiveKeyNums(activeNotes, actor, getWhiteKey) {
			return activeNotes.filter(n => (n.key.isWhiteKey === getWhiteKey) && (n.actor === actor)).map(n => n.key.colourKeyNum);
		}
		const activeWhiteKeysPlayer = getActiveKeyNums(this.piano.activeNotes, Actor.Player, true);
		const activeWhiteKeysModel = getActiveKeyNums(this.piano.activeNotes, Actor.Model, true);
		const activeWhiteKeysBot = getActiveKeyNums(this.piano.activeNotes, Actor.Bot, true);
		const activeBlackKeysPlayer = getActiveKeyNums(this.piano.activeNotes, Actor.Player, false);
		const activeBlackKeysModel = getActiveKeyNums(this.piano.activeNotes, Actor.Model, false);
		const activeBlackKeysBot = getActiveKeyNums(this.piano.activeNotes, Actor.Bot, false);
		
		for (let i = 0; i < this.numWhiteKeys; i++) {
			ctx.fillStyle = PianoCanvas.keyFill.white.inactive;
			
			// Priority: player press > bot press > model press > hover
			if (activeWhiteKeysPlayer.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.white.player;
			} else if (activeWhiteKeysBot.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.white.bot;
			} else if (activeWhiteKeysModel.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.white.model;
			} else if (hoverKeyDefined && this.hoverKey.isWhiteKey && this.hoverKey.colourKeyNum === i) {
				ctx.fillStyle = PianoCanvas.keyFill.white.hover;
			}
			
			const x = this.getXCoordByKey(true, i);
			ctx.fillRect(x, 0, whiteKeyWidth, whiteKeyHeight);
			ctx.strokeRect(x, 0, whiteKeyWidth, whiteKeyHeight);
			
			const hotkey = this.whiteHotkeyMap[i];
			if (hotkey !== undefined) {
				ctx.fillStyle = '#D2D2D2';
				if (hotkey === '.' || hotkey === ',') {
					ctx.font = biggerKeyFont;
				}
				const textWidth = ctx.measureText(hotkey).width;
				ctx.fillText(hotkey, x + (this.whiteKeyWidth / 2) - (textWidth / 2), this.whiteKeyHeight * 0.9);
				ctx.font = keyFont;
			}
		}

		for (let i = 0; i < this.numBlackKeys; i++) {
			ctx.fillStyle = PianoCanvas.keyFill.black.inactive;
			
			// Priority: player press > bot press > model press> hover
			if (activeBlackKeysPlayer.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.black.player;
			} else if (activeBlackKeysBot.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.black.bot;
			} else if (activeBlackKeysModel.includes(i)) {
				ctx.fillStyle = PianoCanvas.keyFill.black.model;
			} else if (hoverKeyDefined && !this.hoverKey.isWhiteKey && this.hoverKey.colourKeyNum === i) {
				ctx.fillStyle = PianoCanvas.keyFill.black.hover;
			}
			
			const x = this.getXCoordByKey(false, i);
			ctx.fillRect(x, 0, blackKeyWidth, blackKeyHeight);
			ctx.strokeRect(x, 0, blackKeyWidth, blackKeyHeight);
			
			const hotkey = this.blackHotkeyMap[i];
			if (hotkey !== undefined) {
				ctx.fillStyle = '#888888';
				const textWidth = ctx.measureText(hotkey).width;
				ctx.fillText(hotkey, x + (this.blackKeyWidth / 2) - (textWidth / 2), this.blackKeyHeight * 0.87);
			}
		}
		
		this.animationQueued = false;
		if (this.piano.activeNotes.length > 0) {
			this.triggerDraw();
		}
	}
}


class PianoKey {
	/*
	Properties
		`keyNum`: 0-indexed absolute number of key on the keyboard, starting from lowest note = 0
		`midiNoteNum`: number in range [0, 127] based on the MIDI specification
		`octave`: the octave that this key belongs in, with the first 3 keys being in octave 0
		`octaveKeyNum`: the key's relative key number (1-indexed) in its octave, e.g. C = 1
		`isWhiteKey`: Boolean for whether the key is white or black
		`colourKeyNum`: 0-indexed key number relative to its colour, e.g. first white key = 0
	*/
	constructor(keyNum, midiNoteNum) {
		this.keyNum = keyNum;
		this.midiNoteNum = midiNoteNum;
		
		this.octave = PianoKey.calcOctave(keyNum);
		this.octaveKeyNum = PianoKey.calcOctaveKeyNum(keyNum);
		this.isWhiteKey = PianoKey.calcIsWhiteKey(keyNum);
		this.colourKeyNum = PianoKey.calcColourKeyNum(keyNum);
		this.keyName = PianoKey.calcKeyName(midiNoteNum);
	}
	
	// Key number of the white keys relative to an octave
	static get whiteKeyNumbers() {
		return [1, 3, 5, 6, 8, 10, 12];
	}
	
	// Key number of the black keys relative to an octave
	static get blackKeyNumbers() {
		return [2, 4, 7, 9, 11];
	}
	
	static get noteNames() {
		return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
	}
	
	static calcOctave(keyNum) {
		return Math.floor((keyNum + 9) / 12);
	}
	
	static calcOctaveKeyNum(keyNum) {
		return ((keyNum + 9) % 12) + 1;
	}
	
	static calcIsWhiteKey(keyNum) {
		const octaveKeyNum = PianoKey.calcOctaveKeyNum(keyNum);
		return PianoKey.whiteKeyNumbers.includes(octaveKeyNum);
	}
	
	static calcColourKeyNum(keyNum) {
		const octave = PianoKey.calcOctave(keyNum);
		const octaveKeyNum = PianoKey.calcOctaveKeyNum(keyNum);
		const isWhiteKey = PianoKey.calcIsWhiteKey(keyNum);
		if (isWhiteKey) {
			return PianoKey.whiteKeyNumbers.indexOf(octaveKeyNum) + (octave * 7) - 5;
		} else {
			return PianoKey.blackKeyNumbers.indexOf(octaveKeyNum) + (octave * 5) - 4;
		}
	}
	
	static calcKeyNumFromColourKeyNum(colourKeyNum, isWhiteKey) {
		if (isWhiteKey) {
			const octave = Math.floor((colourKeyNum + 5) / 7);
			const octaveColourKeyNum = (colourKeyNum + 5) % 7;
			return (octave * 12) - 10 + PianoKey.whiteKeyNumbers[octaveColourKeyNum];
		} else {
			const octave = Math.floor((colourKeyNum + 4) / 5);
			const octaveColourKeyNum = (colourKeyNum + 4) % 5;
			return (octave * 12) - 10 + PianoKey.blackKeyNumbers[octaveColourKeyNum];
		}
	}
	
	static get midiNoteNumMiddleC() {
		return 60;
	}
	
	static calcKeyName(midiNoteNum) {
		const delta = midiNoteNum - PianoKey.midiNoteNumMiddleC;
		const pitchOctave = Math.floor(delta / 12) + 4;
		const index = ((delta % 12) + 12) % 12; // Modulo operation to give non-negative result
		return PianoKey.noteNames[index] + pitchOctave;
	}
}

class PianoKeyMap {
	static get keyMap() {
		return {'q': 15, '2': 16, 'w': 17, '3': 18, 'e': 19, 'r': 20, '5': 21, 't': 22, '6': 23, 'y': 24, '7': 25, 'u': 26, 'i': 27, '9': 28, 'o': 29, '0': 30, 'p': 31, '[': 32, '=': 33, ']': 34, 'a': 35, 'z': 36, 's': 37, 'x': 38, 'c': 39, 'f': 40, 'v': 41, 'g': 42, 'b': 43, 'n': 44, 'j': 45, 'm': 46, 'k': 47, ',': 48, 'l': 49, '.': 50, '/': 51};
	}
}