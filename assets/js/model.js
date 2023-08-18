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
				const genPosition = (basePositionTick + i*48) + "i";
				const notes = s.split(',');
				for (const n of notes) {
					const genNote = Piano.getNoteKeyByNum(parseInt(n) + 1)
					generated.push(new Note(genNote, genPosition));
				}
			}
		}
		return generated;
	}
	
	static historyToQueryString(history, startTick, endTick) {
		const numSixteenthNotes = Math.ceil((endTick - startTick) / 48);
		const orderedNotes = Array.from({ length: numSixteenthNotes }, () => [])
		
		for (const n of history) {
			const p = Tone.Time(n.position).toTicks(); // Position of current note in Ticks
			if (p <= endTick) {
				const d = Math.round((p - startTick) / 48); // Delta between note and startTick in SixteenthNotes
				orderedNotes[d].push(n.noteKey.keyNum - 1);
			}
		}
		
		const queryString = orderedNotes.map(x => x.join(',')).join(';');
		return queryString;
	}
	
	async generateNotes(prevHistory, start, end, buffer) {
		/*
		Arguments:
			`prevHistory`: an array containing recent history of Notes that were played
			`start`: the TransportTime (in BarsQuartersSixteenths) for the start of history period
			`end`: the TransportTime (in BarsQuartersSixteenths) for the end of history period
			`buffer`: the buffer duration (Tone.Time) to add to end of history period
		Note: `start` and `end` define the range of time that is provided and do not correspond to events in `prevHistory`
		*/
		
		const queryString = PianolaModel.historyToQueryString(prevHistory, start, end);
		console.log('Query:', queryString);
		const endpointURI = this.endpoint + new URLSearchParams({notes: queryString});
		const response = await fetch(endpointURI);
		const data = await response.json();
		console.log('Data:', data);
		
		var generated = [];
		if (!data.hasOwnProperty('message')) {
			generated = PianolaModel.parseNotes(data, end + buffer);
		}
		this.noteHistory.push(...generated);
		return generated;
	}
}
