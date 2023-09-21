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

class NoteBar {
	constructor(pianoKey, lastUpdateTime, endTime, relativeX, actor) {
		this.keyNum = pianoKey.keyNum;
		this.isWhiteKey = pianoKey.isWhiteKey;
		this.lastUpdateTime = lastUpdateTime;
		this.endTime = endTime;
		this.relativeX = relativeX;
		this.actor = actor;
		
		this.relativeTop = 1;
		this.relativeBot = 1.005;
	}
	
	static get fill() {
		return {
			player: { white: '#FDFD66', black: '#FFBF00', shadow: 'yellow'},
			model: { white: '#BEFCFF', black: '#5EBBBF', shadow: '#7DF9FF'},
			bot: { white: '#C6FEE2', black: '#59DD9C', shadow: '#79FDBC'}
		};
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
	
	addNoteBar(pianoKey, currTime, endTime, actor) {
		const x = this.piano.pianoCanvas.getXCoordByKey(pianoKey.isWhiteKey, pianoKey.colourKeyNum);
		const relativeX = x / this.canvas.width;
		
		// If this note was previously held, release it before playing the new note
		this.releaseNote(pianoKey.keyNum, currTime);
		const noteBar = new NoteBar(pianoKey, currTime, endTime, relativeX, actor);
		this.activeBars.push(noteBar);
		
		if (!this.animationActive) {
			this.animationActive = true;
			this.triggerAnimation();
		}
	}
	
	releaseNote(keyNum, time) {
		const activeBar = this.activeBars.find(noteBar => noteBar.keyNum === keyNum && (noteBar.endTime > time || noteBar.endTime === -1));
		if (typeof activeBar !== 'undefined') {
			activeBar.endTime = time;
		}
	}
	
	setBPM(bpm) {
		this.bpm = bpm;
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
		
		const shadowBlur = 7;
		ctx.shadowBlur = shadowBlur;
		if (this.activeBars.length > 0) {
			const newActiveBars = [];
			const currTime = new Date();
			const noteLongevity = (6 / (this.bpm / 60)) * 1000; // Number of milliseconds that a note lives on screen (i.e. scrolls from bottom to top)
			
			for (const n of this.activeBars) {
				const yDelta = (currTime - n.lastUpdateTime) / noteLongevity;
				n.relativeTop = Math.max(n.relativeTop - yDelta, 0);
				if (n.endTime <= currTime && n.endTime != -1) {
					n.relativeBot = Math.max(n.relativeBot - yDelta, 0);
				}
				const rectX = n.relativeX * this.canvas.width;
				const rectY = n.relativeTop * this.canvas.height;
				const noteWidth = n.isWhiteKey ? this.piano.pianoCanvas.whiteKeyWidth : this.piano.pianoCanvas.blackKeyWidth;
				const noteHeight = (n.relativeBot - n.relativeTop) * this.canvas.height;
				
				if (n.actor === Actor.Player) {
					ctx.fillStyle = n.isWhiteKey ? NoteBar.fill.player.white : NoteBar.fill.player.black;
					ctx.shadowColor = NoteBar.fill.player.shadow;
				} else if (n.actor === Actor.Bot) {
					ctx.fillStyle = n.isWhiteKey ? NoteBar.fill.bot.white : NoteBar.fill.bot.black;
					ctx.shadowColor = NoteBar.fill.bot.shadow;
				} else if (n.actor === Actor.Model) {
					ctx.fillStyle = n.isWhiteKey ? NoteBar.fill.model.white : NoteBar.fill.model.black;
					ctx.shadowColor = NoteBar.fill.model.shadow;
				}
				ctx.beginPath();
				ctx.roundRect(rectX, rectY, noteWidth, noteHeight, 3);
				ctx.fill();
				
				//if (rectY + noteHeight + shadowBlur > 0) {
				if (n.relativeBot > 0) {
					n.lastUpdateTime = currTime;
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