class PianoRoll {
	constructor(history) {
		this.canvas = PianoRoll.createCanvasElement('164px', '100px', 'pianoRoll');
		this.history = history;
	}
	
	// Left X coordinate of each black key relative to start of an octave
	static get blackKeyX() {
		return [2, 6, 14, 18, 22];
	}
	
	static createCanvasElement(width, height, className) {
		const canvasElement = document.createElement('canvas');
		canvasElement.style.width = width;
		canvasElement.style.height = height;
		if (className) canvasElement.classList.add(className);
		return canvasElement;
	}
	
	draw() {
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = this.canvas.offsetWidth;
		this.canvas.height = this.canvas.offsetHeight;
		
		const noteWidth = 4; // Base width of a note including padding: e.g. whiteKeyWidth = noteWidth - 1, blackKeyWidth = noteWidth - 2
		const noteHeight = 10; // Base height for 1 second, including padding
		const padding = 6;
		ctx.fillStyle = "#1D1D1D";
		ctx.beginPath();
		ctx.roundRect(0, 0, this.canvas.width, this.canvas.height, padding);
		ctx.fill();
		
		// Get the most recent range of notes to be drawn
		const histEndY = (this.history.noteHistory.at(-1).time + this.history.noteHistory.at(-1).duration) * noteHeight;
		const histStartY = Math.max(histEndY - this.canvas.height + padding, -padding);
		
		for (const note of this.history.noteHistory) {
			const x = note.key.isWhiteKey? padding + note.key.colourKeyNum * noteWidth : padding + ((note.key.octave * noteWidth * 7) - (noteWidth * 5)) + PianoRoll.blackKeyX[(note.key.colourKeyNum - 1) % 5];
			const y = note.time * noteHeight - histStartY;
			const width = note.key.isWhiteKey? noteWidth - 1 : noteWidth - 2;
			const height = note.duration * (noteHeight - 3);
			
			if (y + height < 0) continue; // Skip notes that are off canvas
			ctx.fillStyle = note.key.isWhiteKey ? NoteBar.fill[note.actor.name].white : NoteBar.fill[note.actor.name].black;
			ctx.fillRect(x, y, width, height);
		}
	}
}