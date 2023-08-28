class Note {
	constructor(keyNum, position) {
		this.keyNum = keyNum;
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
	
	addNoteBar(pianoKey, currTime, isPlayedByUser) {
		const x = this.piano.pianoCanvas.getXCoordByKey(pianoKey.isWhiteKey, pianoKey.colourKeyNum);
		this.activeBars.push({startTime: currTime, x: x, isWhiteKey: pianoKey.isWhiteKey, isPlayedByUser: isPlayedByUser});
		
		if (!this.animationActive) {
			this.animationActive = true;
			this.triggerAnimation();
		}
	}
	
	triggerAnimation() {
		window.requestAnimationFrame(() => this.draw());
	}
	
	draw() {
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = window.innerWidth;
		this.canvas.height = window.innerHeight - this.piano.pianoCanvas.canvas.height;
		ctx.fillStyle = '#222222';
		ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
		
		// Placeholder square to see if animations are fired
		//ctx.fillStyle = 'red';
		//ctx.fillRect(0, new Date().getMilliseconds() / 10, 10, 10);
		
		const shadowBlur = 10;
		ctx.shadowBlur = shadowBlur;
		if (this.activeBars.length > 0) {
			const newActiveBars = [];
			const currTime = new Date();
			const rectHeight = this.canvas.height / 26;
			
			for (const n of this.activeBars) {
				const rectY = this.canvas.height - ((currTime - n.startTime) * this.canvas.height / 3000);
				const noteWidth = n.isWhiteKey ? this.piano.pianoCanvas.whiteKeyWidth : this.piano.pianoCanvas.blackKeyWidth;
				
				if (n.isPlayedByUser) {
					ctx.fillStyle = n.isWhiteKey ? '#FDFD66' : '#FFBF00';
					ctx.shadowColor = 'yellow';
				} else {
					ctx.fillStyle = n.isWhiteKey ? '#BEFCFF' : '#5EBBBF';
					ctx.shadowColor = '#7DF9FF';
				}
				//ctx.fillRect(n.x, rectY, noteWidth, rectHeight);
				ctx.beginPath();
				ctx.roundRect(n.x, rectY, noteWidth, rectHeight, 3);
				ctx.fill();
				
				if (rectY + rectHeight + shadowBlur > 0) {
					newActiveBars.push(n);
				}
			}
			this.activeBars = newActiveBars;	
		}
		
		if (this.activeBars.length > 0) {
			this.triggerAnimation();
		} else {
			this.animationActive = false;
		}
	}
}