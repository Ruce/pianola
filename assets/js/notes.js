class Note {
	constructor(keyNum, velocity, duration, position) {
		this.keyNum = keyNum;
		this.velocity = velocity;
		this.duration = duration;
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
		this.relativeBot = 1;
	}
	
	static get fill() {
		return {
			player: { white: '#FDFD66', black: '#FFBF00', shadow: 'yellow'},
			model: { white: '#BEFCFF', black: '#5EBBBF', shadow: '#7DF9FF'},
			bot: { white: '#C6FEE2', black: '#59DD9C', shadow: '#79FDBC'}
		};
	}
	
	static get minRelHeight() {
		return 0.005;
	}
}

class NotesCanvas {
	constructor(canvasId, piano) {
		this.piano = piano;
		this.piano.bindNotesCanvas(this);
		
		this.canvas = document.getElementById(canvasId);
		this.activeBars = [];
		
		this.animationActive = false;
		this.glowStart = null;
		this.glowEnd = null;
		this.glowOpacity = 0;
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
	
	startGlow(delayMilliseconds) {
		const glowStart = new Date();
		glowStart.setMilliseconds(glowStart.getMilliseconds() + delayMilliseconds)
		this.glowStart = glowStart;
		this.glowEnd = null;
	}
	
	endGlow() {
		const currDate = new Date();
		this.glowStart = null;
		this.glowEnd = currDate;
	}
	
	updateGlowOpacity() {
		const maxOpacity = 0.4;
		if (this.glowEnd !== null) {
			const delta = new Date() - this.glowEnd;
			this.glowOpacity = Math.min(maxOpacity - (delta / 2000), this.glowOpacity);
		} else if (this.glowStart !== null) {
			const delta = new Date() - this.glowStart;
			this.glowOpacity = Math.max(delta / 3000, this.glowOpacity);
		} else {
			this.glowOpacity = 0;
		}
		this.glowOpacity = Math.min(Math.max(this.glowOpacity, 0), maxOpacity);
		return this.glowOpacity;
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
		
		if (this.updateGlowOpacity() === 0 && this.activeBars.length === 0) {
			this.animationActive = false;
			return;
		}
		
		const newActiveBars = [];
		const currTime = new Date();
		const noteLongevity = (6 / (this.bpm / 60)) * 1000; // Number of milliseconds that a note lives on screen (i.e. scrolls from bottom to top)
		
		if (this.glowOpacity > 0) {
			// Glow effect at the bottom of the canvas
			const glowColor = `rgba(190, 252, 255, ${this.glowOpacity})`;
			const gradient = ctx.createLinearGradient(0, this.canvas.height - 50, 0, this.canvas.height);
			gradient.addColorStop(0, 'transparent');
			gradient.addColorStop(1, glowColor);
			ctx.fillStyle = gradient;
			ctx.fillRect(0, this.canvas.height - 50, this.canvas.width, 50);
		}
		
		const shadowBlur = 7;
		ctx.shadowBlur = shadowBlur;
		for (const n of this.activeBars) {
			const yDelta = (currTime - n.lastUpdateTime) / noteLongevity;
			n.relativeTop = Math.max(n.relativeTop - yDelta, -NoteBar.minRelHeight*2);
			if (n.endTime <= currTime && n.endTime != -1) {
				n.relativeBot = Math.max(n.relativeBot - yDelta, -NoteBar.minRelHeight*2);
			}
			const rectX = n.relativeX * this.canvas.width;
			const rectY = n.relativeTop * this.canvas.height;
			const noteWidth = n.isWhiteKey ? this.piano.pianoCanvas.whiteKeyWidth : this.piano.pianoCanvas.blackKeyWidth;
			const noteHeight = Math.max(n.relativeBot - n.relativeTop, NoteBar.minRelHeight) * this.canvas.height;
			
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
			ctx.roundRect(rectX, rectY, noteWidth, noteHeight, 6);
			ctx.fill();
			
			if (n.relativeBot > -NoteBar.minRelHeight*2) {
				n.lastUpdateTime = currTime;
				newActiveBars.push(n);
			}
		}
		this.activeBars = newActiveBars;
		this.triggerAnimation();
	}
}