import { outputType } from '$lib/Helpers/enums';
import type { VisualizationSpec } from 'vega-embed';

export interface apiResponse {
	outputType: string;
	responseText: string;
	species?: speciesData[];
	vegaSchema?: VisualizationSpec;
	table?: string | {
		latitude: number;
		longitude: number;
		count: number;
	}[];
	html?: string;
}

export interface speciesData {
	id: number;
	concept: string; //species name
	name?: string;
	url?: string;
	taxonomy?: {
		ancestors: {
			name: string;
			rank: string;	
		}[]
		descendants: {
			name: string;
			parent: string;
			rank: string;
		}[]
	};
	rank?: string;
	created_timestamp?: Date;
	depth_meters?: string;
	imaging_type?: string;
	last_updated_timestamp?: Date;
	last_validation?: Date;
	latitude?: number;
	longitude?: number;
	media_type?: string;
	submitter?: string;
	timestamp?: string;
	valid?: string;
	contributors_email?: string;
	altitude?: string;
	oxygen_ml_l?: string;
	pressure_dbar?: string;
	salinity?: string;
	temperature_celsius: string;
	CosineSimilarity: number;
	mr?: {
		region_name: string
	}[];
	x?: number; //x location on image
	y?: number; //y location on image
	width?: number; //width on image
	height?: number; //height on image
}
