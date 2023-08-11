class PianolaModel {
	constructor() {
		this.noteHistory = []; // Array tracking the history of generated notes (ordered from earliest to most recent)
	}
	
	generateNotes(prevHistory, start, end, buffer) {
		/*
		Arguments:
			`prevHistory`: an array containing recent history of notes that were played
			`start`: the TransportTime (in BarsQuartersSixteenths) for the start of history period
			`end`: the TransportTime (in BarsQuartersSixteenths) for the end of history period
			`buffer`: the buffer duration (Tone.Time) to add to end of history period
		Note: `start` and `end` define the range of time that is provided and do not correspond to events in `prevHistory`
		*/
		
		const startTick = Tone.Time(start).toTicks();
		const endTick = Tone.Time(end).toTicks();
		const bufferTick = buffer.toTicks();
		
		const generated = [];
		for (const n of prevHistory) {
			const nTick = Tone.Time(n.position).toTicks();
			const genPosition =  (endTick + bufferTick + (nTick - startTick)) + "i";
			const genNote = n.note;
			generated.push({position: genPosition, note: genNote});
		}
		
		this.noteHistory.push(...generated);
		return generated;
	}
}

//(Tone.Time(transportPosition).toTicks() + Tone.Time("1m").toTicks()) + "i";