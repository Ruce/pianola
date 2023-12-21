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
	
	draw(startTime=0, totalDuration=null) {
		/*
			`startTime`: draw history from this time onwards
			`totalDuration`: scale the drawing to fit this number of seconds into canvas
		*/
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = this.canvas.offsetWidth;
		this.canvas.height = this.canvas.offsetHeight;
		
		const canvasPadding = 6;
		const noteWidth = 4; // Base width of a note including padding: e.g. whiteKeyWidth = noteWidth - 1, blackKeyWidth = noteWidth - 2
		const noteHeight = totalDuration ? Math.max(Math.floor((this.canvas.height - 2*canvasPadding) / totalDuration), 10) : 10; // Base height for 1 second, including padding
		
		ctx.fillStyle = "#1D1D1D";
		ctx.beginPath();
		ctx.roundRect(0, 0, this.canvas.width, this.canvas.height, canvasPadding);
		ctx.fill();
		
		if (this.history.noteHistory.length === 0) return;
		// Get the most recent range of notes to be drawn
		const histEndY = (History.getEndTime(this.history.noteHistory) - startTime) * noteHeight;
		const histStartY = Math.max(histEndY - this.canvas.height + canvasPadding, -canvasPadding);
		
		for (const note of this.history.noteHistory) {
			const x = note.key.isWhiteKey? canvasPadding + note.key.colourKeyNum * noteWidth : canvasPadding + ((note.key.octave * noteWidth * 7) - (noteWidth * 5)) + PianoRoll.blackKeyX[(note.key.colourKeyNum - 1) % 5];
			const y = (note.time - startTime) * noteHeight - histStartY;
			const width = note.key.isWhiteKey? noteWidth - 1 : noteWidth - 2;
			const height = note.duration * (noteHeight - 3);
			
			if (y + height < 0) continue; // Skip notes that are off canvas
			ctx.fillStyle = note.key.isWhiteKey ? NoteBar.fill[note.actor.name].white : NoteBar.fill[note.actor.name].black;
			ctx.fillRect(x, y, width, height);
		}
	}
}