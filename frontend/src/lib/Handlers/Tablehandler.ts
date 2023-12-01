import Table from '$lib/Responses/Table.svelte'
import type { apiResponse } from '$lib/types/responseType';

export function handleTable(target: HTMLElement, jsonResponse: apiResponse): Table {

    return new Table({ target: target, props: { 
		table: jsonResponse.table! as string,
        responseText: jsonResponse.responseText,
		} });

}