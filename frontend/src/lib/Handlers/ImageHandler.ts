import type { apiResponse } from '$lib/types/responseType';
import Image from '$lib/Responses/Image.svelte';
import _fakeRes from '$lib/samples/image.json';
import getImageDetails from '$lib/EventResponseContainer.svelte';

export default function handleImage(target: HTMLElement, res: apiResponse): Image {
	//const fakeRes = _fakeRes as unknown as apiResponse;
	let imageURLs = res.species.map((item) => item.url);
	let imageIDs = res.species.map((item) => item.id);
	let concepts = res.species.map((item) => item.concept);
	let noDuplicateArray = [...new Set(imageURLs)];
	let obj =  new Image({
		target: target,
		props: {
			imageArray: noDuplicateArray,
			imageIDs: imageIDs,
			concepts: concepts,
			naturalTextResponse: res.responseText
		}
	});
	obj.$on('imageSelected', event => {
		console.log('Custom event triggered', event.detail);
		new getImageDetails(event.detail);
	  });

	return obj;
	  
}