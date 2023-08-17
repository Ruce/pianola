class PianolaModel {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.noteHistory = []; // Array tracking the history of generated notes (ordered from earliest to most recent)
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
		
		const startTick = Tone.Time(start).toTicks();
		const endTick = Tone.Time(end).toTicks();
		const bufferTick = buffer.toTicks();
		const generated = [];
		
		const response = await fetch(this.endpoint);
		const data = await response.json();
		
		const notesSlices = data.split(';');
		for (let i=0; i < notesSlices.length; i++) {
			let s = notesSlices[i];
			if (s.length > 0) {
				const genPosition = (endTick + bufferTick + i*48) + "i";
				const notes = s.split(',');
				for (const n of notes) {
					const genNote = Piano.getNoteKeyByNum(parseInt(n) + 1)
					generated.push(new Note(genNote, genPosition));
				}
			}
		}
			
		
		this.noteHistory.push(...generated);
		return generated;
	}
}
