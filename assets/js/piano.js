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
	
	static calcOctave(keyNum) {
		return Math.floor((keyNum + 9) / 12);
	}
	
	static calcOctaveKeyNum(keyNum) {
		return ((keyNum + 9) % 12) + 1;
	}
	
	static calcIsWhiteKey(keyNum) {
		const octaveKeyNum = PianoKey.calcOctaveKeyNum(keyNum);
		return Piano.whiteKeyNumbers.includes(octaveKeyNum);
	}
	
	static calcColourKeyNum(keyNum) {
		const octave = PianoKey.calcOctave(keyNum);
		const octaveKeyNum = PianoKey.calcOctaveKeyNum(keyNum);
		const isWhiteKey = PianoKey.calcIsWhiteKey(keyNum);
		if (isWhiteKey) {
			return Piano.whiteKeyNumbers.indexOf(octaveKeyNum) + (octave * 7) - 5;
		} else {
			return Piano.blackKeyNumbers.indexOf(octaveKeyNum) + (octave * 5) - 4;
		}
	}
	
	static get midiNoteNumMiddleC() {
		return 60;
	}
	
	static calcKeyName(midiNoteNum) {
		const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
		
		const delta = midiNoteNum - PianoKey.midiNoteNumMiddleC;
		const pitchOctave = Math.floor(delta / 12) + 4;
		const index = ((delta % 12) + 12) % 12; // Modulo operation to give non-negative result
		return noteNames[index] + pitchOctave;
	}
}

class Piano {
	constructor(canvasId, octaves, model) {
		if (octaves < 1 || octaves > 7) {
			throw new RangeError("The number of octaves must be between 1 and 7");
		}
		this.octaves = octaves;
		this.numWhiteKeys = 7 * this.octaves + 3; // 3 additional keys before and after main octaves
		this.numBlackKeys = 5 * this.octaves + 1; // 1 additional key in the 0th octave
		this.pianoKeys = Piano.createPianoKeys(this.octaves);
		
		this.model = model;
		this.sampler = this.initialiseSampler();
		this.toneStarted = false;
		this.isCallingModel = false;
		this.lastActivity = new Date();
		this.noteHistory = [];
		this.activeKeys = [];
		
		this.canvas = document.getElementById(canvasId);
		this.animationQueued = false;
		this.triggerDraw();
		this.canvas.addEventListener('mousedown', this.keyboardClicked.bind(this));
		this.canvas.addEventListener('mousemove', this.mouseMoveKeyboard.bind(this));
		this.canvas.addEventListener('mouseout', this.mouseOutKeyboard.bind(this));
		
		this.hoverKey = null;
		this.prevHoverKey = null;
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
	
	static get keyFill() {
		return {
			white: {inactive: '#FEFEFE', hover: '#FFF9D7', playerPress: '#FEF3B0', botPress: '#A6DDDF'},
			black: {inactive: '#595959', hover: '#D0AD40', playerPress: '#C09200', botPress: '#478C8F'}
		};
	}
	
	// Key number of the white keys relative to an octave
	static get whiteKeyNumbers() {
		return [1, 3, 5, 6, 8, 10, 12];
	}
	
	// Key number of the black keys relative to an octave
	static get blackKeyNumbers() {
		return [2, 4, 7, 9, 11];
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
	
	static createPianoKeys(octaves) {
		const numKeys = (7 * octaves + 3) + (5 * octaves + 1); // White keys + black keys, and extra keys outside of main octaves
		const lowestMidiNote = PianoKey.midiNoteNumMiddleC - (Math.floor(octaves / 2) * 12) - 3; // Calculate lowest note from middle C
		const pianoKeys = [];
		for (let i = 0; i < numKeys; i++) {
			pianoKeys.push(new PianoKey(i, lowestMidiNote + i));
		}
		return pianoKeys;
	}
	
	getKeyByCoord(clientX, clientY) {
		const canvasRect = this.canvas.getBoundingClientRect();
		const x = clientX - canvasRect.left;
		const y = clientY - canvasRect.top;
		
		const octaveWidth = this.whiteKeyWidth * 7;
		const o = Math.floor((x + (this.whiteKeyWidth * 5)) / octaveWidth); // Current octave
		const deltaX = x - ((o-1) * octaveWidth) - (2 * this.whiteKeyWidth); // x position relative to octave
		
		if (y > this.blackKeyHeight) {
			// Must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = Piano.whiteKeyNumbers[n];
			const keyNum = (octaveKeyNum - 1) + o*12 - 9;
			return this.pianoKeys[keyNum];
		} else if (o === this.octaves + 1) {
			// Only highest C is in the highest octave
			const keyNum = o * 12 - 9;
			return this.pianoKeys[keyNum];
		} else {
			for (let i=0; i < Piano.blackKeyPos.length; i++) {
				if (o === 0 && i < 4) {
					// 0th octave does not have first 4 black keys
					continue;
				}
				const pos = Piano.blackKeyPos[i];
				const blackKeyLeft = this.whiteKeyWidth * pos;
				const blackKeyRight = blackKeyLeft + this.blackKeyWidth;
				// Except for octave 0, which only has 1 black key
				if (deltaX >= blackKeyLeft && deltaX <= blackKeyRight) {
					const octaveKeyNum = Piano.blackKeyNumbers[i];
					const keyNum = (octaveKeyNum - 1) + o*12 - 9;
					return this.pianoKeys[keyNum];
				}
			}
			// Not a black key, therefore must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = Piano.whiteKeyNumbers[n];
			const keyNum = (octaveKeyNum - 1) + o*12 - 9;
			return this.pianoKeys[keyNum];
		}
	}
	
	getXCoordByKey(isWhiteKey, colourKeyNum) {
		if (isWhiteKey) {
			return this.whiteKeyWidth * colourKeyNum;
		} else {
			const k = (colourKeyNum + 4) % 5; // Index of the 5 black keys in `blackKeyPos`
			const o = Math.floor((colourKeyNum-1) / 5); // Current octave (first full octave is index 0, unlike PianoKey convention)
			return this.whiteKeyWidth * (Piano.blackKeyPos[k] + o*7 + 2);
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
	
	playNote(pianoKey, time, transportPosition=Tone.Transport.position) {
		const currTime = new Date();
		this.sampler.triggerAttackRelease(pianoKey.keyName, 0.25, time);
		
		const isPlayer = (typeof time === 'undefined');
		const endTime = new Date(currTime);
		endTime.setMilliseconds(endTime.getMilliseconds() + 200); // Key lights up for 200 milliseconds
		this.activeKeys.push({key: pianoKey, endTime: endTime, isPlayer: isPlayer});
		this.noteHistory.push(new Note(pianoKey.keyNum, transportPosition));
		
		this.triggerDraw();
		// Draw note on canvas
		if (typeof this.notesCanvas !== 'undefined') {
			this.notesCanvas.addNoteBar(pianoKey, currTime, isPlayer);
		}
		return transportPosition;
	}
	
	scheduleNote(pianoKey, triggerTime) {
		Tone.Transport.scheduleOnce((time) => this.playNote(pianoKey, time, triggerTime), triggerTime);
	}
	
	async callModel() {
		//const start = typeof this.prevCallEnd === 'undefined' ? 0 : this.prevCallEnd;
		const end = Tone.Time(Tone.Transport.position).toTicks();
		const start = Math.max(0, end - Tone.Time('8').toTicks());
		this.prevCallEnd = end;
		
		const recentHistory = Note.getRecentHistory(this.noteHistory, start);
		const buffer = Tone.Time("1").toTicks()
		const generated = await this.model.generateNotes(recentHistory, start, end, buffer);
		
		// Check if the model is still active (i.e. hasn't been stopped) before scheduling notes
		if (this.isCallingModel) {
			for (const note of generated) {
				this.scheduleNote(this.pianoKeys[note.keyNum], note.position);
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
	
	startCallModel() {
		if (!this.isCallingModel) {
			this.isCallingModel = true;
			this.callModelIntervalId = setInterval(() => this.callModel(), 1000);
			this.checkActivityIntervalId = setInterval(() => this.checkActivity(), 5000);
		}
	}
	
	stopCallModel() {
		if (this.isCallingModel && typeof this.callModelIntervalId !== 'undefined') {
			this.isCallingModel = false;
			clearInterval(this.callModelIntervalId);
		}
		if (typeof this.checkActivityIntervalId !== 'undefined') {
			clearInterval(this.checkActivityIntervalId);
		}
	}
	
	keyboardClicked(event) {
		this.startTone();
		this.lastActivity = new Date();
		this.startCallModel();
		
		globalMouseDown = true;
		const clickedKey = this.getKeyByCoord(event.clientX, event.clientY);
		this.playNote(clickedKey);
	}
	
	mouseMoveKeyboard(event) {
		this.hoverKey = this.getKeyByCoord(event.clientX, event.clientY);
		if (this.prevHoverKey === null || this.prevHoverKey.keyNum !== this.hoverKey.keyNum) {
			// Newly moused over key
			this.triggerDraw();
			if (globalMouseDown) {
				this.playNote(this.hoverKey);
			}
			this.prevHoverKey = this.hoverKey;
		}
	}
	
	mouseOutKeyboard(event) {
		this.triggerDraw();
		this.hoverKey = null;
		this.prevHoverKey = null;
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
		this.canvas.height = this.canvas.width * Piano.keyboardRatio;
		ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

		this.whiteKeyWidth = this.canvas.width / this.numWhiteKeys;
		this.whiteKeyHeight = this.canvas.height;
		this.blackKeyWidth = this.whiteKeyWidth * Piano.blackKeyWidthRatio;
		this.blackKeyHeight = this.whiteKeyHeight * Piano.blackKeyHeightRatio;
		const [whiteKeyWidth, whiteKeyHeight, blackKeyWidth, blackKeyHeight] = [this.whiteKeyWidth, this.whiteKeyHeight, this.blackKeyWidth, this.blackKeyHeight];
		
		// Remove expired keys and get the colourKeyNums for active keys
		this.activeKeys = this.activeKeys.filter((k) => k.endTime >= new Date());
		function getActiveKeyNums(activeKeys, getWhiteKey, getPlayerKey) {
			return activeKeys.filter((k) => (k.key.isWhiteKey === getWhiteKey) && (k.isPlayer === getPlayerKey)).map((k) => k.key.colourKeyNum);
		}
		const activeWhiteKeysPlayer = getActiveKeyNums(this.activeKeys, true, true);
		const activeWhiteKeysBot = getActiveKeyNums(this.activeKeys, true, false);
		const activeBlackKeysPlayer = getActiveKeyNums(this.activeKeys, false, true);
		const activeBlackKeysBot = getActiveKeyNums(this.activeKeys, false, false);
		
		for (let i = 0; i < this.numWhiteKeys; i++) {
			ctx.fillStyle = Piano.keyFill.white.inactive;
			
			// Priority: player press > bot press > hover
			if (activeWhiteKeysPlayer.includes(i)) {
				ctx.fillStyle = Piano.keyFill.white.playerPress;
			} else if (activeWhiteKeysBot.includes(i)) {
				ctx.fillStyle = Piano.keyFill.white.botPress;
			} else if (hoverKeyDefined && this.hoverKey.isWhiteKey && this.hoverKey.colourKeyNum === i) {
				ctx.fillStyle = Piano.keyFill.white.hover;
			}
			
			const x = this.getXCoordByKey(true, i);
			ctx.fillRect(x, 0, whiteKeyWidth, whiteKeyHeight);
			ctx.strokeRect(x, 0, whiteKeyWidth, whiteKeyHeight);
		}

		for (let i = 0; i < this.numBlackKeys; i++) {
			ctx.fillStyle = Piano.keyFill.black.inactive;
			
			// Priority: player press > bot press > hover
			if (activeBlackKeysPlayer.includes(i)) {
				ctx.fillStyle = Piano.keyFill.black.playerPress;
			} else if (activeBlackKeysBot.includes(i)) { // Prioritise player keys
				ctx.fillStyle = Piano.keyFill.black.botPress;
			} else if (hoverKeyDefined && !this.hoverKey.isWhiteKey && this.hoverKey.colourKeyNum === i) {
				ctx.fillStyle = Piano.keyFill.black.hover;
			}
			
			const x = this.getXCoordByKey(false, i);
			ctx.fillRect(x, 0, blackKeyWidth, blackKeyHeight);
			ctx.strokeRect(x, 0, blackKeyWidth, blackKeyHeight);
		}
		
		this.animationQueued = false;
		if (this.activeKeys.length > 0) {
			this.triggerDraw();
		}
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
			release: 0.5,
			volume: -6
		}).toDestination();
		
		return sampler;
	}
	
	bindNotesCanvas(notesCanvas) {
		this.notesCanvas = notesCanvas;
	}
}