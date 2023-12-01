import HeatMap from '$lib/Responses/HeatMap.svelte';
import type { apiResponse } from '$lib/types/responseType';

export default function handleHeatMap(target: HTMLElement, jsonResponse: apiResponse): HeatMap {
	let positionData: {x: number, y: number, z: number}[] = [];
	
	if(jsonResponse.species !== undefined) {
		for(let i = 0; i < jsonResponse.species.length; i++) {
			if(jsonResponse.species[i].longitude !== undefined && jsonResponse.species[i].latitude !== undefined) {
				positionData.push({
					x: jsonResponse.species[i].latitude!,
					y: jsonResponse.species[i].longitude!,
					z: 1
				});
			} else {
				console.error("[TREY] lat or long not found in species data");
			}
			
		}
	} else if (jsonResponse.table !== undefined) {
		let table = jsonResponse.table as {latitude: number, longitude: number, count: number}[];
		for(let entry of table) {
			positionData.push({
				x: entry.latitude,
				y: entry.longitude,
				z: entry.count,
			});
		}
	}

	return new HeatMap({ target: target, props: {
		responseText: jsonResponse.responseText,
		positionData: positionData,
	} });
}
