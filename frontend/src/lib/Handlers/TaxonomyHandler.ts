import Taxonomy from '$lib/Responses/Taxonomy.svelte';
export default function handleTaxonomy(target: HTMLElement, jsonResponse: any): Taxonomy {

	return new Taxonomy({ target: target, props: { 
		speciesArray: jsonResponse.species as any[],
		responseText: jsonResponse.responseText
		} });
}
