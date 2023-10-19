class Note {
	constructor(pianoKey, velocity, duration, time, actor, scheduleId, isRewind=false) {
		this.key = pianoKey;
		this.velocity = velocity;
		this.duration = duration;
		this.time = time;
		this.actor = actor;
		this.scheduleId = scheduleId;
		this.isRewind = isRewind;
	}
	
	static getRecentHistory(history, startTime) {
		// Returns events in history that happened on or after `startTime`
		// `history` must be an ordered list of events from first to most recent
		const recentHistory = [];
		for (let i = history.length - 1; i >= 0; i--) {
			const h = history[i];
			if (h.time >= startTime) {
				recentHistory.unshift(h);
			} else {
				break;
			}
		}
		return recentHistory;
	}
	
	static removeHistory(history, startTime) {
		// Remove events in history that happened on or after `startTime`
		// `history` must be an ordered list of events from first to most recent
		// Returns new history array without altering original `history`
		for (let i = history.length - 1; i >= 0; i--) {
			if (history[i].time < startTime) {
				return history.slice(0, i+1);
			}
		}
		return [];
	}
	
	getPosition(bpm) {
		const beats = this.time * bpm / 60000;
		return `0:${beats}`
	}
}

class NoteBar {
	constructor(pianoKey, startTime, duration, bpm, relativeX, actor, isRewind) {
		this.keyNum = pianoKey.keyNum;
		this.isWhiteKey = pianoKey.isWhiteKey;
		this.startTime = startTime;
		this.duration = duration;
		this.bpm = bpm;
		this.relativeX = relativeX;
		this.actor = actor;
		this.isRewind = isRewind;
		
		this.relativeTop = 1;
		this.relativeBot = 1;
		this.lastUpdateTime = startTime;
	}
	
	static get fill() {
		return {
			player: { white: '#FDFD66', black: '#FFBF00', shadow: 'yellow'},
			model: { white: '#BEFCFF', black: '#5EBBBF', shadow: '#7DF9FF'},
			bot: { white: '#AFFED7', black: '#59DD9C', shadow: '#79FDBC'}
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
		this.lastAnimationCheck = null;
		this.frameCount = 0;
		this.enableShadows = true;
		this.enableRoundRect = true;
		
		this.glowStart = null;
		this.glowEnd = null;
		this.glowOpacity = 0;
		this.triggerAnimation();
	}
	
	addNoteBar(pianoKey, startTime, duration, bpm, actor, isRewind) {
		const x = this.piano.pianoCanvas.getXCoordByKey(pianoKey.isWhiteKey, pianoKey.colourKeyNum);
		const relativeX = x / this.canvas.width;
		
		const noteBar = new NoteBar(pianoKey, startTime, duration, bpm, relativeX, actor, isRewind);
		this.activeBars.push(noteBar);
		
		if (!this.animationActive) {
			this.animationActive = true;
			this.triggerAnimation();
		}
	}
	
	releaseNote(keyNum, finalDuration) {
		const activeBar = this.activeBars.find(n => n.keyNum === keyNum && (n.duration === - 1 || n.duration > new Date() - n.startTime)); // Note that is held down or hasn't finished playing
		if (typeof activeBar !== 'undefined') {
			activeBar.duration = finalDuration + 0.02; // Add a small delta to increase height of note bar so it transitions smoothly upon release
		}
	}
	
	triggerAnimation() {
		window.requestAnimationFrame(timestamp => this.draw(timestamp));
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
			// Fade out glow
			const delta = new Date() - this.glowEnd;
			this.glowOpacity = Math.min(maxOpacity - (delta / 2000), this.glowOpacity);
		} else if (this.glowStart !== null) {
			// Fade in glow
			const delta = new Date() - this.glowStart;
			this.glowOpacity = Math.max(delta / 3000, this.glowOpacity);
		} else {
			this.glowOpacity = 0;
		}
		this.glowOpacity = Math.min(Math.max(this.glowOpacity, 0), maxOpacity);
		return this.glowOpacity;
	}
	
	draw(timestamp) {
		const ctx = this.canvas.getContext('2d');
		this.canvas.width = this.canvas.offsetWidth;
		this.canvas.height = this.canvas.offsetHeight;
		
		ctx.fillStyle = '#2A2A2A';
		ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
		
		// Check if there is anything to animate
		if (this.updateGlowOpacity() === 0 && this.activeBars.length === 0) {
			this.animationActive = false;
			this.lastAnimationCheck = null;
			this.frameCount = 0;
			this.enableShadows = true;
			this.enableRoundRect = true;
			return;
		}
		
		// Debug placeholder square to see if animations are fired
		//ctx.fillStyle = 'red';
		//ctx.fillRect(0, new Date().getMilliseconds() / 10, 10, 10);
		
		// Check the current fps and disable effects if slow
		this.frameCount++;
		if (this.lastAnimationCheck === null) {
			this.lastAnimationCheck = timestamp;
		} else {
			const elapsed = timestamp - this.lastAnimationCheck;
			if (elapsed > 1000) {
				const fps = this.frameCount / (elapsed / 1000);
				if (fps < 50) {
					if (this.enableShadows) {
						this.enableShadows = false;
					} else {
						this.enableRoundRect = false;
					}
				}
				this.frameCount = 0;
				this.lastAnimationCheck = timestamp;
			}
		}
		
		// Glow effect at the bottom of the canvas
		if (this.glowOpacity > 0) {
			const glowColor = `rgba(190, 252, 255, ${this.glowOpacity})`;
			const gradient = ctx.createLinearGradient(0, this.canvas.height - 50, 0, this.canvas.height);
			gradient.addColorStop(0, 'transparent');
			gradient.addColorStop(1, glowColor);
			ctx.fillStyle = gradient;
			ctx.fillRect(0, this.canvas.height - 50, this.canvas.width, 50);
		}
		
		// Draw note bars
		const newActiveBars = [];
		const currTime = new Date();
		const noteSpeedFactor = 6;
		const noteLongevity = (noteSpeedFactor / (this.piano.bpm / 60)) * 1000; // Number of milliseconds that a note lives on screen (i.e. scrolls from bottom to top)
		if (this.enableShadows) ctx.shadowBlur = 6;
		for (const n of this.activeBars) {
			const yDelta = (currTime - n.lastUpdateTime) / noteLongevity;
			n.relativeTop -= yDelta;
			const relativeHeight = Math.max((n.duration * (n.bpm / 60) / noteSpeedFactor) - 0.006, NoteBar.minRelHeight); // Add a small gap between consecutive notes
			const relativeBot = n.duration === -1 ? 1 + NoteBar.minRelHeight*2 : n.relativeTop + relativeHeight;
			
			const rectX = n.relativeX * this.canvas.width + 1;
			const rectY = Math.max(n.relativeTop, -NoteBar.minRelHeight*2) * this.canvas.height;
			const noteWidth = n.isWhiteKey ? this.piano.pianoCanvas.whiteKeyWidth - 2 : this.piano.pianoCanvas.blackKeyWidth;
			const noteHeight = Math.max((relativeBot * this.canvas.height) - rectY, 0);
			
			ctx.globalAlpha = n.isRewind ? 0.4 : 1;
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
			if (this.enableRoundRect) {
				ctx.beginPath();
				ctx.roundRect(rectX, rectY, noteWidth, noteHeight, 7);
				ctx.fill();
			} else {
				ctx.fillRect(rectX, rectY, noteWidth, noteHeight);
			}
			
			if (relativeBot > -NoteBar.minRelHeight*2) {
				n.lastUpdateTime = currTime;
				newActiveBars.push(n);
			}
		}
		this.activeBars = newActiveBars;
		this.triggerAnimation();
	}
}