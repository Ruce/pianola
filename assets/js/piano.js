class Piano {
	constructor(canvasId, octaves, model) {
		if (octaves < 1 || octaves > 7) {
			throw new RangeError("The number of octaves must be between 1 and 7");
		}
		this.octaves = octaves;
		this.pianoKeys = Piano.createPianoKeys(this.octaves);
		this.pianoCanvas = new PianoCanvas(this, canvasId);
		this.sampler = this.initialiseSampler();
		this.model = model;
		
		this.toneStarted = false;
		this.isCallingModel = false;
		this.awaitingPlayerInput = true;
		this.activeKeys = [];
		this.noteHistory = [];
		this.lastActivity = new Date();
		
		this.bufferBeats = 4;
		this.bufferTicks = Tone.Time(`0:${this.bufferBeats}`).toTicks();
		this.historyWindowBeats = 36;
		this.defaultBPM = 88;
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
	
	keyPressed(keyNum) {
		this.startTone();
		this.lastActivity = new Date();
		if (this.awaitingPlayerInput) {
			this.setBPM(this.defaultBPM);
			this.seedInputListener();
		}
		this.playNote(this.pianoKeys[keyNum], 0.7, -1, Actor.Player);
	}
	
	playNote(pianoKey, velocity, duration, actor, time, transportPosition=Tone.Transport.position) {
		const currTime = new Date();
		this.noteHistory.push(new Note(pianoKey.keyNum, velocity, duration, transportPosition));
		
		let endTime = -1;
		if (duration === -1) {
			// Note is being held down
			this.sampler.triggerAttack(pianoKey.keyName, time, velocity);
		} else {
			const durationMilliseconds = ((duration * 60 * 1000) / (this.bpm * 4)) * 0.9 - 15; // Shorten duration slightly to add a gap between notes
			const triggerDuration = (durationMilliseconds / 1000) + 0.15; // Add a short delay to the end of the sound (in seconds)
			this.sampler.triggerAttackRelease(pianoKey.keyName, triggerDuration, time, velocity);
			
			endTime = new Date(currTime);
			endTime.setMilliseconds(endTime.getMilliseconds() + durationMilliseconds); // Key lights up for 150 milliseconds
		}
		
		// Check if pianoKey is already active, and if so overwrite its `endTime`
		const activeKey = this.activeKeys.find(obj => obj.key === pianoKey);
		if (typeof activeKey === 'undefined') {
			this.activeKeys.push({key: pianoKey, endTime: endTime, actor: actor});
		} else {
			activeKey.endTime = endTime;
		}
		
		// Draw note on canvases
		this.pianoCanvas.triggerDraw();
		this.notesCanvas.addNoteBar(pianoKey, currTime, endTime, actor);
		
		return transportPosition;
	}
	
	releaseNote(pianoKey) {
		// Check if `pianoKey` is in the `activeKeys` array, and if so release the note
		const activeKey = this.activeKeys.find(obj => obj.key === pianoKey);
		const keyIndex = this.activeKeys.indexOf(activeKey);
		if (keyIndex > -1) {
			this.sampler.triggerRelease(pianoKey.keyName, Tone.now() + 0.1);
			this.activeKeys.splice(keyIndex, 1);
			this.notesCanvas.releaseNote(pianoKey.keyNum, new Date());
		}
	}
	
	releaseAllNotes() {
		const heldKeys = this.activeKeys.filter(obj => obj.endTime === -1);
		for (const heldKey of heldKeys) {
			this.releaseNote(heldKey.key);
		}
	}
	
	scheduleNote(pianoKey, velocity, duration, triggerTime, actor) {
		Tone.Transport.scheduleOnce((time) => this.playNote(pianoKey, velocity, duration, actor, time, triggerTime), triggerTime);
	}
	
	async callModel(customHistory) {
		// From previous time the model was called, add buffer duration to get new interval for querying
		this.callModelEnd += this.bufferTicks;
		
		 // Since buffer will be added to the history that is sent to the model, shorten the "played history" window by `bufferBeats`
		const recentBeats = this.historyWindowBeats - this.bufferBeats;
		const start = Math.max(0, this.callModelEnd - Tone.Time(`0:${recentBeats}`).toTicks() + 1);
		
		const history = typeof customHistory !== 'undefined' ? customHistory : Note.getRecentHistory(this.noteHistory, start);	
		const generated = await this.model.generateNotes(history, start, this.callModelEnd, this.bufferTicks);
		
		// Check if the model is still active (i.e. hasn't been stopped) before scheduling notes
		if (this.isCallingModel) {
			for (const note of generated) {
				this.scheduleNote(this.pianoKeys[note.keyNum], note.velocity, note.duration, note.position, Actor.Model);
			}
		}
	}
	
	checkActivity() {
		const timeout = 5 * 60 * 1000; // in milliseconds
		if (new Date() - this.lastActivity > timeout) {
			console.log('No user activity, stopping model...');
			this.stopCallModel();
		}
	}
	
	startCallModel(customHistory) {
		if (!this.isCallingModel) {
			this.isCallingModel = true;
			this.callModel(customHistory); // Call model immediately, since setInterval first triggers function after the delay
			this.callModelIntervalId = setInterval(() => this.callModel(), this.callModelSeconds*1000);
			this.checkActivityIntervalId = setInterval(() => this.checkActivity(), 5000);
			this.notesCanvas.startGlow(this.callModelSeconds*1000);
		}
	}
	
	stopCallModel() {
		this.isCallingModel = false;
		this.awaitingPlayerInput = true;
		this.lastSeedInputPosition = null;
		this.callModelInitial = null;
		this.notesCanvas.endGlow();
		
		this.noteHistory = [];
		this.model.noteHistory = [];
		Tone.Transport.cancel();
		Tone.Transport.stop();
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
		if (typeof this.lastSeedInputPosition === 'undefined' || this.lastSeedInputPosition === null) {
			const listenerElement = document.getElementById("listener");
			listenerElement.style.visibility = "visible";
			this.listenerIntervalId = setInterval(this.seedInputAwaiter.bind(this), 100);
		}
		this.lastSeedInputPosition = Tone.Transport.position;
	}
	
	seedInputAwaiter() {
		if (new Date() - this.lastActivity > 2000 && !globalMouseDown) {
			clearInterval(this.listenerIntervalId);
			
			// Rewind the transport schedule to the last of the seed input so that the history fed to the model is seamless
			// Subtract twice the buffer duration from the last seed input because callModel and generateNotes each adds a buffer
			this.callModelEnd = Tone.Time(this.lastSeedInputPosition).toTicks() - (2 * this.bufferTicks);
			const newPosition = Tone.Time(this.lastSeedInputPosition).toSeconds() - Tone.Time(`0:${this.bufferBeats}`).toSeconds();
			Tone.Transport.position = newPosition;
			
			this.awaitingPlayerInput = false;
			this.lastSeedInputPosition = null;
			this.startCallModel();
			document.getElementById("listener").style.visibility = "hidden";
		}
	}
	
	playExample(data, bpm) {
		this.stopCallModel();
		this.awaitingPlayerInput = false;
		this.setBPM(bpm);
		this.startTone();
		this.lastActivity = new Date();
		
		let lastNoteTick = 0;
		const startPosition = Tone.Time(Tone.Transport.position).toTicks();
		const notes = PianolaModel.queryStringToNotes(data, startPosition);
		for (const note of notes) {
			this.scheduleNote(this.pianoKeys[note.keyNum], note.velocity, note.duration, note.position, Actor.Bot);
			lastNoteTick = Tone.Time(note.position).toTicks();
		}
		
		// Set the end position for the first model call to 2*bufferTicks before last note because callModel and generateNotes each adds a buffer
		// Then schedule startCallModel to trigger one bufferTick before the last note
		this.callModelEnd = lastNoteTick - (2 * this.bufferTicks);
		Tone.Transport.scheduleOnce(() => this.startCallModel(notes), Math.max(0, (this.callModelEnd + this.bufferTicks)) + "i");
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
		this.callModelSeconds = this.bufferBeats / bpm * 60;
		if (typeof this.notesCanvas !== 'undefined') {
			this.notesCanvas.setBPM(bpm);
		}
	}
	
	startTone() {
		if (!this.toneStarted) { 
			Tone.start().then(() => {
				Tone.Transport.start();
			});
			this.toneStarted = true;
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
		
		this.hoverKey = null;
		this.prevHoverKey = null;
		this.animationQueued = false;
		
		this.canvas = document.getElementById(canvasId);
		this.canvas.addEventListener('mousedown', this.mouseDownKeyboard.bind(this));
		this.canvas.addEventListener('mousemove', this.mouseMoveKeyboard.bind(this));
		this.canvas.addEventListener('mouseup', this.mouseOutKeyboard.bind(this));
		this.canvas.addEventListener('mouseout', this.mouseOutKeyboard.bind(this));
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
	
	mouseOutKeyboard(event) {
		this.hoverKey = null;
		this.prevHoverKey = null;
		this.piano.releaseAllNotes();
		this.triggerDraw();
		this.piano.lastActivity = new Date();
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
		
		// Remove expired keys and get the colourKeyNums for active keys
		this.piano.activeKeys = this.piano.activeKeys.filter((k) => k.endTime >= new Date() || k.endTime === -1);
		function getActiveKeyNums(activeKeys, actor, getWhiteKey) {
			return activeKeys.filter((k) => (k.key.isWhiteKey === getWhiteKey) && (k.actor === actor)).map((k) => k.key.colourKeyNum);
		}
		const activeWhiteKeysPlayer = getActiveKeyNums(this.piano.activeKeys, Actor.Player, true);
		const activeWhiteKeysModel = getActiveKeyNums(this.piano.activeKeys, Actor.Model, true);
		const activeWhiteKeysBot = getActiveKeyNums(this.piano.activeKeys, Actor.Bot, true);
		const activeBlackKeysPlayer = getActiveKeyNums(this.piano.activeKeys, Actor.Player, false);
		const activeBlackKeysModel = getActiveKeyNums(this.piano.activeKeys, Actor.Model, false);
		const activeBlackKeysBot = getActiveKeyNums(this.piano.activeKeys, Actor.Bot, false);
		
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
		}
		
		this.animationQueued = false;
		if (this.piano.activeKeys.length > 0) {
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
