class History {
	constructor(bpm, name) {
		this.bpm = bpm;
		this.name = name;
		this.start = new Date();
		this.noteHistory = [];
		this.lastSeedNoteTime = 0;
		this.numRewinds = 0;
		this.isNew = false; // Tracks whether any new (non-rewind) notes have been played
	}
	
	static getRecentHistory(noteHistory, startTime) {
		// Returns events in history that happened on or after `startTime`
		// `noteHistory` must be an ordered list of events from first to most recent
		const recentHistory = [];
		for (let i = noteHistory.length - 1; i >= 0; i--) {
			const h = noteHistory[i];
			if (h.time >= startTime) {
				recentHistory.unshift(h);
			} else {
				break;
			}
		}
		return recentHistory;
	}
	
	static removeHistory(noteHistory, startTime) {
		// Remove events in history that happened on or after `startTime`
		// `noteHistory` must be an ordered list of events from first to most recent
		// Returns new array without altering original `noteHistory`
		for (let i = noteHistory.length - 1; i >= 0; i--) {
			if (noteHistory[i].time < startTime) {
				return noteHistory.slice(0, i+1);
			}
		}
		return [];
	}
	
	add(note) {
		if (!note.isRewind) this.isNew = true;
		
		// Ensure that note is added in chronological order within noteHistory
		let i = this.noteHistory.length - 1;
		while (i >= 0) {
			if (this.noteHistory[i].time <= note.time) {
				break;
			}
			i--;
		}
		this.noteHistory.splice(i+1, 0, note);
	}
	
	copy() {
		// Deep copy this History with new copies of Notes that have isRewind = true
		const newHistory = new History(this.bpm, this.name);
		newHistory.lastSeedNoteTime = this.lastSeedNoteTime;
		for (const note of this.noteHistory) {
			newHistory.add(new Note(note.key, note.velocity, note.duration, note.time, note.actor, null, true));
		}
		return newHistory;
	}
}
