class PianolaModel {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.noteHistory = []; // Array tracking the history of generated notes (ordered from earliest to most recent)
	}
	
	static queryStringToNotes(data, basePositionTick) {
		const generated = [];
		const timesteps = data.split(';');
		for (let i=0; i < timesteps.length; i++) {
			const step = timesteps[i];
			const genPosition = (basePositionTick + i*48) + "i";
			for (let i = 0; i < step.length; i += 6) {
				const note = step.slice(i, i + 6);
				const noteNum = parseInt(note.slice(0, 2));
				const velocity = (parseInt(note.slice(2, 4)) + 1) / 100;
				const duration = parseInt(note.slice(4, 6));
				generated.push(new Note(noteNum, velocity, duration, genPosition));
			}
		}
		return generated;
	}
	
	static historyToQueryString(history, startTick, endTick) {
		function toPaddedNumber(number) {
			return number.toString().padStart(2, '0');
		}
		
		const numSixteenthNotes = Math.floor((endTick - startTick) / 48) + 1;		
		const orderedNotes = Array.from({ length: numSixteenthNotes }, () => [])
		
		for (const n of history) {
			const p = Tone.Time(n.position).toTicks(); // Position of current note in Ticks
			if (p <= endTick) {
				const t = Math.floor((p - startTick) / 48); // Delta between note and startTick in SixteenthNotes
				const velocity = Math.max(Math.min(Math.round((n.velocity * 100) - 1), 99), 0); // Velocity scaled to be 0-indexed, between 0 and 99
				const duration = Math.max(Math.min(n.duration, 99), 0);
				const noteStr = `${toPaddedNumber(n.keyNum)}${toPaddedNumber(velocity)}${toPaddedNumber(duration)}`;
				orderedNotes[t].push(noteStr);
			}
		}
		
		const queryString = orderedNotes.map(x => x.join('')).join(';');
		return queryString;
	}
	
	async queryModel(queryString) {
		const endpointURI = this.endpoint + new URLSearchParams({notes: queryString, timesteps: 16});
		try {
			const response = await fetch(endpointURI);
			const data = await response.json();
			return data;
		} catch (error) {
			console.log('Error connecting to endpoint:', error);
		}
	}
	
	async connectToModel(callback) {
		const data = await this.queryModel(";");
		console.log('Data:', data);
		
		if (typeof data !== 'undefined' && !data.hasOwnProperty('message')) {
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
			const newBasePosition = end + buffer + 48; // New notes will start 1 timestep (i.e. 48 ticks) after the end+buffer window
			generated = PianolaModel.queryStringToNotes(data, newBasePosition);
		}
		this.noteHistory.push(...generated);
		return generated;
	}
}
