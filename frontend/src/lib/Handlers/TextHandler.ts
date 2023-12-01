import type { apiResponse } from '$lib/types/responseType';
import Text from '$lib/Responses/Text.svelte';
import _fakeRes from '$lib/samples/text.json';

export default function handleText(target: HTMLElement, content: string): Text {

	return new Text({
		target: target,
		props: {
			text: content,
		}
	});
}
