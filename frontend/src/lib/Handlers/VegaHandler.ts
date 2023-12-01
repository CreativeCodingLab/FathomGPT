import VegaVisual from '$lib/Responses/VegaVisual.svelte';
import type { apiResponse } from '$lib/types/responseType';
import type { VisualizationSpec } from 'vega-embed';

export default function handleVega(target: HTMLElement, jsonResponse: apiResponse): VegaVisual {

	return new VegaVisual({
		target: target,
        props: {
            spec: jsonResponse.vegaSchema as VisualizationSpec,
            responseText: jsonResponse.responseText
        }
	});
}