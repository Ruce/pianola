class Note {
	constructor(noteKey, position) {
		this.noteKey = noteKey;
		this.position = position;
	}
	
	static getRecentHistory(history, startTick) {
		const recentHistory = [];
		for (let i = history.length - 1; i >= 0; i--) {
			const h = history[i];
			if (Tone.Time(h.position).toTicks() >= startTick) {
				recentHistory.unshift(h);
			} else {
				break;
			}
		}
		return recentHistory;
	}
}

class NotesCanvas {
	constructor(canvasId, piano) {
		this.piano = piano;
		this.piano.bindNotesCanvas(this);
		
		this.canvas = document.getElementById(canvasId);
		this.activeBars = [];
		
		this.animationActive = false;
		this.triggerAnimation();
	}
	
	addNoteBar(noteKey, currTime) {
		const x = this.piano.getXCoordByKey(noteKey.isWhiteKey, noteKey.colourKeyNum);
		this.activeBars.push({startTime: currTime, x: x, isWhiteKey: noteKey.isWhiteKey});
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
		
		ctx.shadowBlur = 10;
		if (this.activeBars.length > 0) {
			const newActiveBars = [];
			const currTime = new Date();
			for (const n of this.activeBars) {
				const rectY = this.canvas.height - ((currTime - n.startTime) * this.canvas.height / 4000);
				const rectHeight = this.canvas.height / 25;
				const noteWidth = n.isWhiteKey ? this.piano.whiteKeyWidth : this.piano.blackKeyWidth;
				
				ctx.fillStyle = n.isWhiteKey ? '#FDFD66' : '#FFDF00';
				ctx.shadowColor = 'yellow';
				//ctx.fillRect(n.x, rectY, noteWidth, rectHeight);
				ctx.beginPath();
				ctx.roundRect(n.x, rectY, noteWidth, rectHeight, 3);
				ctx.fill();
				
				if (rectY + rectHeight > 0) {
					newActiveBars.push(n);
				}
			}
			this.activeBars = newActiveBars;	
		}
		this.animationActive = false;
		if (this.activeBars.length > 0) { this.triggerAnimation(); }
	}
}