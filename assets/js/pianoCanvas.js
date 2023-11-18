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
			black: {inactive: '#4D4D4D', hover: '#D0AD40', player: '#C09200', model: '#478C8F', bot: '#2C6E4E'}
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
		if (globalMouseDown && this.hoverKey !== null) {
			this.piano.releaseNote(this.hoverKey);
			this.piano.lastActivity = new Date();
		}
		this.hoverKey = null;
		this.prevHoverKey = null;
		this.triggerDraw();
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
		this.canvas.width = this.canvas.offsetWidth;
		this.canvas.height = this.canvas.offsetHeight;
		
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
		this.piano.activeNotes = this.piano.activeNotes.filter(n => n.duration === -1 || n.time + n.duration > Tone.Transport.seconds);
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