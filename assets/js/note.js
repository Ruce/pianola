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
	
	getPosition(bpm) {
		const beats = this.time * bpm / 60000;
		return `0:${beats}`
	}
}
