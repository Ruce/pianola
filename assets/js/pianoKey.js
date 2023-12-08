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
	
	static createPianoKeys(octaves) {
		const numKeys = (7 * octaves + 3) + (5 * octaves + 1); // White keys + black keys, and extra keys outside of main octaves
		const lowestMidiNote = PianoKey.midiNoteNumMiddleC - (Math.floor(octaves / 2) * 12) - 3; // Calculate lowest note from middle C
		const pianoKeys = [];
		for (let i = 0; i < numKeys; i++) {
			pianoKeys.push(new PianoKey(i, lowestMidiNote + i));
		}
		return pianoKeys;
	}
}

class PianoKeyMap {
	static get whiteKeyMap() {
		return {'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4, 'y': 5, 'u': 6, 'i': 7, 'o': 8, 'p': 9, '[': 10, ']': 11, 'z': 12, 'x': 13, 'c': 14, 'v': 15, 'b': 16, 'n': 17, 'm': 18, ',': 19, '.': 20, '/': 21};
	}
	
	static get blackKeyMap() {
		return {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '0': 9, '-': 10, '=': 11, 'a': 12, 's': 13, 'd': 14, 'f': 15, 'g': 16, 'h': 17, 'j': 18, 'k': 19, 'l': 20, ';': 21, "'": 22};
	}
	
	static getKeyNum(keyChar, shift) {
		// N.B. Returns keyNums that may be out of range of the piano keyboard if shift is negative or very high
		const isWhiteKey = keyChar in PianoKeyMap.whiteKeyMap;
		const isBlackKey = keyChar in PianoKeyMap.blackKeyMap;
		if (isWhiteKey && isBlackKey) {
			throw new Error('Invalid key maps');
		} else if (isWhiteKey) {
			const colourKeyNum = PianoKeyMap.whiteKeyMap[keyChar];
			return PianoKey.calcKeyNumFromColourKeyNum(colourKeyNum + shift, true);
		} else if (isBlackKey) {
			// Check whether this is a valid black key
			const blackIdx = PianoKeyMap.blackKeyMap[keyChar];
			const adjWhiteKeyNum = PianoKey.calcKeyNumFromColourKeyNum(blackIdx + shift, true); // Get the keyNum of the white key to the right of this black key
			
			// Check if the colour of the key before adjWhiteKey: if it is white, then keyChar is invalid (i.e. a non-existent black key based on `shift`)
			if (PianoKey.calcIsWhiteKey(adjWhiteKeyNum - 1)) {
				return null;
			} else {
				return adjWhiteKeyNum - 1;
			}
		} else {
			return null;
		}
	}
	
	static getKeyMap(shift, numKeys) {
		const keyMap = {};
		for (const k of Object.keys(PianoKeyMap.whiteKeyMap)) {
			const keyNum = PianoKeyMap.getKeyNum(k, shift);
			if (keyNum !== null && keyNum >= 0 && keyNum < numKeys) {
				keyMap[k] = keyNum;
			}
		}
		for (const k of Object.keys(PianoKeyMap.blackKeyMap)) {
			const keyNum = PianoKeyMap.getKeyNum(k, shift);
			if (keyNum !== null && keyNum >= 0 && keyNum < numKeys) {
				keyMap[k] = keyNum;
			}
		}
		return keyMap;
	}
}