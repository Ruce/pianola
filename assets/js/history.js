class History {
	constructor(bpm, name) {
		this.bpm = bpm;
		this.name = name;
		this.start = new Date();
		this.numRewinds = 0;
		this.noteHistory = [];
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
	
	add(note) {
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
}
