class Note {
	constructor(keyNum, position) {
		this.keyNum = keyNum;
		this.position = position;
	}
	
	static getRecentHistory(history, startTick) {
		// Returns events in history that happened on or after `startTick`
		// `history` must be an ordered list of events from first to most recent
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
	
	static get barFill() {
		return {
			player: { white: '#FDFD66', black: '#FFBF00', shadow: 'yellow'},
			model: { white: '#BEFCFF', black: '#5EBBBF', shadow: '#7DF9FF'},
			bot: { white: '#C6FEE2', black: '#59DD9C', shadow: '#79FDBC'}
		};
	}
	
	addNoteBar(pianoKey, currTime, actor) {
		const x = this.piano.pianoCanvas.getXCoordByKey(pianoKey.isWhiteKey, pianoKey.colourKeyNum);
		this.activeBars.push({startTime: currTime, x: x, isWhiteKey: pianoKey.isWhiteKey, actor: actor});
		
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
		
		// Debug placeholder square to see if animations are fired
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
				
				if (n.actor === Actor.Player) {
					ctx.fillStyle = n.isWhiteKey ? NotesCanvas.barFill.player.white : NotesCanvas.barFill.player.black;
					ctx.shadowColor = NotesCanvas.barFill.player.shadow;
				} else if (n.actor === Actor.Bot) {
					ctx.fillStyle = n.isWhiteKey ? NotesCanvas.barFill.bot.white : NotesCanvas.barFill.bot.black;
					ctx.shadowColor = NotesCanvas.barFill.bot.shadow;
				} else if (n.actor === Actor.Model) {
					ctx.fillStyle = n.isWhiteKey ? NotesCanvas.barFill.model.white : NotesCanvas.barFill.model.black;
					ctx.shadowColor = NotesCanvas.barFill.model.shadow;
				}
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