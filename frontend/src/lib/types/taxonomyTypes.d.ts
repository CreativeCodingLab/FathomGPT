import { taxonomyLevel } from '$lib/Helpers/enums';

export interface taxonomystructure {
	concept: string;
	rank: string;
	taxnomy: {
		ancestors: {
			name: string;
			rank: string;	
		}[]
		descendents: {
			name: string;
			parent: string;
			rank: string;
		}[]
	}
}
