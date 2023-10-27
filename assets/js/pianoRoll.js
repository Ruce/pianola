class PianoRoll {
	constructor() {
		this.canvas = PianoRoll.createCanvasElement('164px', '92px', 'pianoRoll');
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
	
	draw(history) {
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = this.canvas.offsetWidth;
		this.canvas.height = this.canvas.offsetHeight;
		
		const padding = 6;
		ctx.fillStyle = "#2A2A2A";
		ctx.beginPath();
		ctx.roundRect(0, 0, this.canvas.width, this.canvas.height, padding);
		ctx.fill();
		
		for (const note of history.noteHistory) {
			const x = note.key.isWhiteKey? padding + note.key.colourKeyNum * 4 : padding + ((note.key.octave * 28) - 20) + PianoRoll.blackKeyX[(note.key.colourKeyNum - 1) % 5];
			const y = padding + note.time * 15;
			const width = note.key.isWhiteKey? 3 : 2;
			const height = note.duration * 12;
			ctx.fillStyle = note.key.isWhiteKey ? NoteBar.fill[note.actor.name].white : NoteBar.fill[note.actor.name].black;
			ctx.fillRect(x, y, width, height);
		}
	}
}