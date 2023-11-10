class Actor {
	static Player = new Actor('player');
	static Bot = new Actor('bot');
	static Model = new Actor('model');
	
	static get Actors() {
		// Return an ordered list of all types of Actor
		return [Actor.Player, Actor.Bot, Actor.Model];
	}
	
	constructor(name) {
		this.name = name;
	}
	
	toString() {
		return `Actor.${this.name}`;
	}
}