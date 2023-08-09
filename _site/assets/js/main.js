class PianoKey {
	/*
	Properties
		`keyNum`: 1-indexed absolute number of key on the keyboard, starting from lowest note = 1
		`octave`: the octave that this key belongs in, with the first 3 keys being in octave 0
		`octaveKeyNum`: the key's relative key number (1-indexed) in its octave, e.g. C = 1
		`isWhiteKey`: Boolean for whether the key is white or black
		`colourKeyNum`: 0-indexed key number relative to its colour, e.g. first white key = 0
	*/
	constructor(keyNum, octave, octaveKeyNum, isWhiteKey) {
		this.keyNum = keyNum;
		this.octave = octave;
		this.octaveKeyNum = octaveKeyNum;
		this.isWhiteKey = isWhiteKey;
		this.colourKeyNum = this.calcColourKeyNum();
	}
	
	calcColourKeyNum() {
		if (this.isWhiteKey) {
			return Piano.whiteKeyNumbers.indexOf(this.octaveKeyNum) + this.octave*7 - 5;
		} else {
			return Piano.blackKeyNumbers.indexOf(this.octaveKeyNum) + this.octave*5 - 4;
		}
	}
}

class Piano {
	constructor(canvasId, octaves) {
		this.octaves = octaves;
		this.whiteKeys = 7 * this.octaves + 3; // 3 additional keys before and after main octaves
		this.blackKeys = 5 * this.octaves + 1; // 1 additional key in the 0th octave
		
		this.canvas = document.getElementById(canvasId);
		this.drawKeyboard();
		this.canvas.addEventListener('mousedown', this.keyboardClicked.bind(this));
		this.canvas.addEventListener('mousemove', this.mouseMoveKeyboard.bind(this));
		this.canvas.addEventListener('mouseout', this.mouseOutKeyboard.bind(this));
		
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
			'white': {'inactive': '#FEFEFE', 'active': '#FEF3B0'},
			'black': {'inactive': '#595959', 'active': '#C09200'}
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
	
	getKeyByCoord(x, y) {
		const octaveWidth = this.whiteKeyWidth * 7;
		const o = Math.floor((x + (this.whiteKeyWidth * 5)) / octaveWidth); // Current octave
		const deltaX = x - ((o-1) * octaveWidth) - (2 * this.whiteKeyWidth); // x position relative to octave
		
		if (y > this.blackKeyHeight) {
			// Must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = Piano.whiteKeyNumbers[n];
			const keyNum = octaveKeyNum + o*12 - 9;
			return new PianoKey(keyNum, o, octaveKeyNum, true);
		} else if (o === this.octaves + 1) {
			// Only highest C is in the highest octave
			return new PianoKey(o * 12 - 8, o, 1, true);
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
					const keyNum = octaveKeyNum + o*12 - 9;
					return new PianoKey(keyNum, o, octaveKeyNum, false);
				}
			}
			// Not a black key, therefore must be a white key
			const n = Math.floor(deltaX / this.whiteKeyWidth);
			const octaveKeyNum = Piano.whiteKeyNumbers[n];
			const keyNum = octaveKeyNum + o*12 - 9;
			return new PianoKey(keyNum, o, octaveKeyNum, true);
		}
	}
	
	keyboardClicked(event) {
		const canvasRect = this.canvas.getBoundingClientRect();
		const clickX = event.clientX - canvasRect.left;
		const clickY = event.clientY - canvasRect.top;
		const clickedKey = this.getKeyByCoord(clickX, clickY)
		console.log(clickedKey.keyNum, clickedKey.octave, clickedKey.octaveKeyNum, clickedKey.isWhiteKey);
	}
	
	mouseMoveKeyboard(event) {
		const hoverKey = this.getKeyByCoord(event.clientX, event.clientY);
		if (this.prevHoverKey === null || this.prevHoverKey.keyNum !== hoverKey.keyNum) {
			this.drawKeyboard(hoverKey);
			this.prevHoverKey = hoverKey;
		}
	}
	
	mouseOutKeyboard(event) {
		this.drawKeyboard();
		this.prevHoverKey = null;
	}
	
	drawKeyboard(hoverKey) {
		const hoverKeyDefined = (typeof hoverKey !== 'undefined');
		
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = window.innerWidth;
		this.canvas.height = this.canvas.width * Piano.keyboardRatio;
		ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

		this.whiteKeyWidth = this.canvas.width / this.whiteKeys;
		this.whiteKeyHeight = this.canvas.height;
		this.blackKeyWidth = this.whiteKeyWidth * Piano.blackKeyWidthRatio;
		this.blackKeyHeight = this.whiteKeyHeight * Piano.blackKeyHeightRatio;
		const [whiteKeyWidth, whiteKeyHeight, blackKeyWidth, blackKeyHeight] = [this.whiteKeyWidth, this.whiteKeyHeight, this.blackKeyWidth, this.blackKeyHeight];
		
		for (let i = 0; i < this.whiteKeys; i++) {
			ctx.fillStyle = Piano.keyFill.white.inactive;
			if (hoverKeyDefined && hoverKey.isWhiteKey && hoverKey.colourKeyNum === i) {
				ctx.fillStyle = Piano.keyFill.white.active;
			}
			const x = i * whiteKeyWidth;
			ctx.fillRect(x, 0, whiteKeyWidth, whiteKeyHeight);
			ctx.strokeRect(x, 0, whiteKeyWidth, whiteKeyHeight);
		}

		for (let i = 0; i < this.blackKeys; i++) {
			ctx.fillStyle = Piano.keyFill.black.inactive;
			if (hoverKeyDefined && !hoverKey.isWhiteKey && hoverKey.colourKeyNum === i) {
				ctx.fillStyle = Piano.keyFill.black.active;
			}
			
			const k = (i+4) % 5; // Index of the 5 black keys in `blackKeyPos`
			const o = Math.floor((i-1) / 5); // Current octave
			const x = whiteKeyWidth * (Piano.blackKeyPos[k] + o*7 + 2);
			ctx.fillRect(x, 0, blackKeyWidth, blackKeyHeight);
			ctx.strokeRect(x, 0, blackKeyWidth, blackKeyHeight);
		}
	}
}

const octaves = 5;
function initialisePiano() {
	const piano = new Piano('pianoCanvas', octaves);

	var resizeTimeout = false;
	const resizeDelay = 250;
	window.onresize = function () {
		clearTimeout(resizeTimeout);
		resizeTimeout = setTimeout(piano.drawKeyboard.bind(piano), resizeDelay);
	}
}

document.addEventListener("DOMContentLoaded", initialisePiano);