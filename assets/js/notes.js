class NotesCanvas {
	constructor(canvasId, piano) {
		this.piano = piano;
		this.piano.bindNotesCanvas(this);
		
		this.canvas = document.getElementById(canvasId);
		this.activeNotes = [];
		
		this.animationActive = false;
		this.triggerAnimation();
	}
	
	addNote(noteKey, currTime) {
		const x = this.piano.getXCoordByKey(noteKey.isWhiteKey, noteKey.colourKeyNum);
		const noteWidth = noteKey.isWhiteKey ? this.piano.whiteKeyWidth : this.piano.blackKeyWidth;
		this.activeNotes.push({startTime: currTime, x: x, width: noteWidth});
		this.triggerAnimation();
	}
	
	triggerAnimation() {
		if (!this.animationActive) {
			window.requestAnimationFrame(() => this.draw());
		}
	}
	
	draw() {
		this.animationActive = true;
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = window.innerWidth;
		this.canvas.height = window.innerHeight - piano.canvas.height;
		ctx.fillStyle = '#222222';
		ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
		
		// Placeholder square to see if animations are fired
		//ctx.fillStyle = 'red';
		//ctx.fillRect(0, new Date().getMilliseconds() / 10, 10, 10);
		
		ctx.fillStyle = 'yellow';
		if (this.activeNotes.length > 0) {
			const newActiveNotes = [];
			const currTime = new Date();
			for (const n of this.activeNotes) {
				const rectY = this.canvas.height - ((currTime - n.startTime) * this.canvas.height / 4000);
				const rectHeight = this.canvas.height / 30;
				ctx.fillRect(n.x, rectY, n.width, rectHeight);
				
				if (rectY + rectHeight > 0) {
					newActiveNotes.push(n);
				}
			}
			this.activeNotes = newActiveNotes;	
		}
		this.animationActive = false;
		if (this.activeNotes.length > 0) { this.triggerAnimation(); }
	}
}