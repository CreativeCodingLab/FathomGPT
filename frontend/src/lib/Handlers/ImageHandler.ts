import type { apiResponse } from '$lib/types/responseType';
import Image from '$lib/Responses/Image.svelte';
import _fakeRes from '$lib/samples/image.json';

export default function handleImage(target: HTMLElement, res: apiResponse): Image {
	//const fakeRes = _fakeRes as unknown as apiResponse;
	let imageURLs = res.species.map((item) => item.url);
	let noDuplicateArray = [...new Set(imageURLs)];
	return new Image({
		target: target,
		props: {
			imageArray: noDuplicateArray,
			naturalTextResponse: res.responseText
		}
	});
}