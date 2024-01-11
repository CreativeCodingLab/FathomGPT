import type { apiResponse, speciesData } from '$lib/types/responseType';
import Image from '$lib/Responses/Image.svelte';
import _fakeRes from '$lib/samples/image.json';

export default function handleImage(target: HTMLElement, res: apiResponse): Image {
	//const fakeRes = _fakeRes as unknown as apiResponse;
	let species: Array<speciesData> = (res.species as Array<speciesData>).filter((v, i, a) => a.findIndex(t => (t.url === v.url)) === i);
	return new Image({
		target: target,
		props: {
			species: species,
			naturalTextResponse: res.responseText
		}
	});
}