import Plotly from '$lib/Responses/Plotly.svelte'
import type { apiResponse } from '$lib/types/responseType';

export function handleVisualization(target: HTMLElement, jsonResponse: apiResponse): Plotly {

    return new Plotly({ target: target, props: { 
		generate_html: jsonResponse.html! as string,
        naturalTextResponse: jsonResponse.responseText,
		} });

}