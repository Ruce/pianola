class PianolaModel {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.noteHistory = []; // Array tracking the history of generated notes (ordered from earliest to most recent)
	}
	
	static parseNotes(data, basePositionTick) {
		const generated = [];
		const notesSlices = data.split(';');
		for (let i=0; i < notesSlices.length; i++) {
			let s = notesSlices[i];
			if (s.length > 0) {
				const genPosition = (basePositionTick + (i+1)*48) + "i";
				const notes = s.split(',');
				for (const n of notes) {
					generated.push(new Note(parseInt(n), genPosition));
				}
			}
		}
		return generated;
	}
	
	static historyToQueryString(history, startTick, endTick) {
		const numSixteenthNotes = Math.floor((endTick - startTick) / 48) + 1;		
		const orderedNotes = Array.from({ length: numSixteenthNotes }, () => [])
		
		for (const n of history) {
			const p = Tone.Time(n.position).toTicks(); // Position of current note in Ticks
			if (p <= endTick) {
				const d = Math.floor((p - startTick) / 48); // Delta between note and startTick in SixteenthNotes
				orderedNotes[d].push(n.keyNum);
			}
		}
		
		const queryString = orderedNotes.map(x => x.join(',')).join(';');
		return queryString;
	}
	
	async queryModel(queryString) {
		const endpointURI = this.endpoint + new URLSearchParams({notes: queryString, timesteps: 8});
		const response = await fetch(endpointURI);
		const data = await response.json();
		return data;
	}
	
	async connectToModel(callback) {
		const data = await this.queryModel(";");
		console.log('Data:', data);
		
		if (!data.hasOwnProperty('message')) {
			// Expected response received, trigger callback
			callback();
		} else {
			// Wait a short while before trying to connect to model again
			setTimeout(this.connectToModel.bind(this, callback), 500);
		}
	}
	
	async generateNotes(prevHistory, start, end, buffer) {
		/*
		Arguments:
			`prevHistory`: an array containing recent history of Notes that were played
			`start`: the TransportTime (in Ticks) for the start of history period
			`end`: the TransportTime (in Ticks) for the end of history period
			`buffer`: the buffer duration (Tone.Time in Ticks) to add to end of history period
		Note: `start` and `end` define the range of time that is provided and do not correspond to events in `prevHistory`
		*/
		
		// Get "history" from buffer (i.e. notes queued up to be played) and combine with prevHistory (i.e. notes that have been played)
		const recentNoteHistory = Note.getRecentHistory(this.noteHistory, end);
		recentNoteHistory.push(...prevHistory);
		
		const queryString = PianolaModel.historyToQueryString(recentNoteHistory, start, end+buffer);
		console.log('Query:', queryString);
		const data = await this.queryModel(queryString);
		console.log('Data:', data);
		
		var generated = [];
		if (!data.hasOwnProperty('message')) {
			generated = PianolaModel.parseNotes(data, end + buffer);
		}
		this.noteHistory.push(...generated);
		return generated;
	}
}
