class PianolaModel {
	constructor(endpoint) {
		this.endpoint = endpoint;
		this.lastActivity = null;
		this.lastQuery = null;
		this.keepAliveIntervalId = setInterval(() => this.keepAlive(), 10000);
	}
		
	static queryStringToNotes(data, start, bpm) {
		const s = 60 / (bpm * 4); // Length of a sixteenth note in seconds
		
		const generated = [];
		const timesteps = data.split(';');
		for (let i=0; i < timesteps.length; i++) {
			const step = timesteps[i];
			const time = start + i*s;
			for (let i = 0; i < step.length; i += 6) {
				const note = step.slice(i, i + 6);
				const noteNum = parseInt(note.slice(0, 2));
				const velocity = (parseInt(note.slice(2, 4)) + 1) / 100;
				const duration = parseInt(note.slice(4, 6)) * s;
				generated.push({keyNum: noteNum, velocity: velocity, duration: duration, time: time});
			}
		}
		return generated;
	}
	
	static historyToQueryString(history, start, end, bpm) {
		function toPaddedNumber(number) {
			return number.toString().padStart(2, '0');
		}
		
		const s = 60 / (bpm * 4); // Length of a sixteenth note in seconds
		const numSixteenthNotes = Math.floor((end - start) / s) + 1;
		const orderedNotes = Array.from({ length: numSixteenthNotes }, () => [])
		
		for (const n of history) {
			if (n.time <= end) {
				const t = Math.floor((n.time - start) / s); // Delta between note and startTick in SixteenthNotes
				const velocity = Math.max(Math.min(Math.round((n.velocity * 100) - 1), 99), 0); // Velocity scaled to be 0-indexed, between 0 and 99
				const duration = Math.max(Math.min(Math.round(n.duration / s), 99), 1);
				const noteStr = `${toPaddedNumber(n.key.keyNum)}${toPaddedNumber(velocity)}${toPaddedNumber(duration)}`;
				orderedNotes[t].push(noteStr);
			}
		}
		
		const queryString = orderedNotes.map(x => x.join('')).join(';');
		return queryString;
	}
	
	async queryModel(queryString, timesteps) {
		this.lastQuery = new Date();
		const endpointURI = this.endpoint + new URLSearchParams({notes: queryString, timesteps: timesteps});
		try {
			const response = await fetch(endpointURI);
			const data = await response.json();
			return data;
		} catch (error) {
			console.log('Error connecting to endpoint:', error);
		}
	}
	
	async connectToModel(callback) {
		this.lastActivity = new Date();
		const data = await this.queryModel(";", 1);
		console.log(new Date().toISOString(), `Connected to model [${data}]`);
		
		if (typeof data !== 'undefined' && !data.hasOwnProperty('message')) {
			// Expected response received, trigger callback
			callback();
		} else {
			// Wait a short while before trying to connect to model again
			setTimeout(this.connectToModel.bind(this, callback), 500);
		}
	}
	
	async generateNotes(history, start, end, bpm, timesteps) {
		/*
		Arguments:
			`history`: an array containing recent history of Notes that were played
			`start`: the TransportTime (in seconds) for the start of history period
			`end`: the TransportTime (in seconds) for the end of history period
			`bpm`: beats per minute that the notes in `history` were played at
			`timesteps`: number of timesteps to generate
		*/
		this.lastActivity = new Date();
		const queryString = PianolaModel.historyToQueryString(history, start, end, bpm);
		console.log(new Date().toISOString(), 'Query:', queryString);
		const data = await this.queryModel(queryString, timesteps);
		console.log(new Date().toISOString(), 'Data:', data);
		
		var generated = [];
		if (!data.hasOwnProperty('message')) {
			const generatedStart = end + (60 / (bpm * 4)); // New notes will start a sixteenth note after `end` time
			generated = PianolaModel.queryStringToNotes(data, generatedStart, bpm);
		}
		return generated
	}
	
	async keepAlive() {
		const currTime = new Date();
		const activityTimeout = 300000;
		const queryInterval = 30000;
		// If the model has been active in `activityTimeout` window and no queries have been made in the last `queryInterval`, query the model to keep it alive
		if (this.lastActivity !== null && currTime - this.lastActivity < activityTimeout && this.lastQuery !== null && currTime - this.lastQuery > queryInterval) {
			const data = await this.queryModel(";", 1);
		}
	}
}
