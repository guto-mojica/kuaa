// Shared scene/film fixtures used by all three direction shells.
// Multi-film attribution is the v1.0 shape, so every scene has a film_id.

window.FILMS = [
  { id: 'jeca',      title: 'Jeca Tatu',                year: 1959, director: 'Mazzaropi',                 country: 'BR', scenes: 412, runtime: 96  },
  { id: 'limite',    title: 'Limite',                   year: 1931, director: 'Mário Peixoto',             country: 'BR', scenes: 187, runtime: 114 },
  { id: 'rio40',     title: 'Rio, 40 Graus',            year: 1955, director: 'Nelson Pereira dos Santos', country: 'BR', scenes: 263, runtime: 100 },
  { id: 'cangaceiro',title: 'O Cangaceiro',             year: 1953, director: 'Lima Barreto',              country: 'BR', scenes: 298, runtime: 105 },
  { id: 'aruanda',   title: 'Aruanda',                  year: 1960, director: 'Linduarte Noronha',         country: 'BR', scenes:  94, runtime:  21 },
  { id: 'pagador',   title: 'O Pagador de Promessas',   year: 1962, director: 'Anselmo Duarte',            country: 'BR', scenes: 334, runtime:  98 },
];

// Sample search results for query "duas pessoas conversando ao ar livre"
// Mixed across films to demonstrate multi-film attribution.
window.RESULTS = [
  { id: 1,  film:'jeca',       kf:'keyframes/kf-11-mustache.jpg',    tc:'00:21:58:08', cena:111, score:0.873, desc:'A man in a checkered shirt speaks with a figure across a field of dry grass, gestures held mid-sentence.', tags:['exterior','duas-pessoas','dia','rural-field'] },
  { id: 2,  film:'pagador',    kf:'keyframes/kf-03-horse.jpg',       tc:'00:01:57:18', cena:3,   score:0.851, desc:'A rider on a pale horse pauses at the field\u2019s edge, addressing labourers in the distance.', tags:['exterior','horse-rider','rural-field','dia'] },
  { id: 3,  film:'jeca',       kf:'keyframes/kf-12-mustache2.jpg',   tc:'00:22:15:18', cena:115, score:0.847, desc:'Continuation of the field exchange; the second man enters frame, hat pushed back.', tags:['exterior','duas-pessoas','sertão'] },
  { id: 4,  film:'cangaceiro', kf:'keyframes/kf-06-women-hut.jpg',   tc:'00:03:08:06', cena:6,   score:0.821, desc:'Two women in traditional clothing speak in front of a thatched roof; one carries a basket.', tags:['exterior','duas-pessoas','rural','dia'] },
  { id: 5,  film:'aruanda',    kf:'keyframes/kf-05-man-cow.jpg',     tc:'00:02:51:21', cena:5,   score:0.793, desc:'A man tends to his cart and oxen at sunrise; a second figure approaches from the rear.', tags:['exterior','labor','duas-pessoas'] },
  { id: 6,  film:'rio40',      kf:'keyframes/kf-04-cow.jpg',         tc:'00:02:34:05', cena:4,   score:0.781, desc:'A cow stands in a field, a wooden wagon and tree under a cloudy sky\u2014two figures distant.', tags:['exterior','cow','wagon','dia'] },
  { id: 7,  film:'jeca',       kf:'keyframes/kf-02-fence.jpg',       tc:'00:01:45:15', cena:2,   score:0.762, desc:'A man in a white shirt rests on a wooden fence with a dog on his lap; another figure off-screen.', tags:['exterior','fence','dog','dia'] },
  { id: 8,  film:'pagador',    kf:'keyframes/kf-07-woman-pot.jpg',   tc:'00:03:18:15', cena:7,   score:0.748, desc:'A woman in white holds a vessel under a thatched roof, looking out at the open landscape.', tags:['exterior','mountain','dia'] },
  { id: 9,  film:'limite',     kf:'keyframes/kf-08-woman-dark.jpg',  tc:'00:14:08:22', cena:48,  score:0.731, desc:'A woman stands in a dim interior, holding a basket; light filters from the doorway.', tags:['interior','baixa-luz'] },
];

// Visual rhymes / cross-film visual echo group (signature feature)
window.RHYMES = {
  anchor: { film:'jeca', kf:'keyframes/kf-03-horse.jpg', tc:'00:01:57:18', cena:3, desc:'Rider in middle distance, lush field, mountain horizon.' },
  echoes: [
    { film:'cangaceiro', kf:'keyframes/kf-05-man-cow.jpg',   tc:'00:42:11:07', cena:217, score:0.94, note:'figure + livestock, mid-frame, dawn light' },
    { film:'aruanda',    kf:'keyframes/kf-07-woman-pot.jpg', tc:'00:08:22:13', cena:34,  score:0.89, note:'solitary figure, distant horizon line' },
    { film:'rio40',      kf:'keyframes/kf-04-cow.jpg',       tc:'00:54:03:18', cena:182, score:0.86, note:'rural still life, cloudy sky pressure' },
  ]
};

window.TAGS = ['exterior','duas-pessoas','dia','rural-field','horse-rider','sertão','interior','baixa-luz','noite','close-up','title-card','crowd','farm','wagon','cow'];
