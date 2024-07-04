from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# Input article
article_pl = """Gospodarstwo Powrót panicza — Spotkanie się pierwsze w pokoiku, drugie u stołu — Ważna Sędziego nauka o grzeczności — Podkomorzego uwagi polityczne nad modami — Początek sporu o Kusego i Sokoła — Żale Wojskiego — Ostatni Woźny Trybunału — Rzut oka na ówczesny stan polityczny Litwy i Europy

Litwo! Ojczyzno moja! ty jesteś jak zdrowie:
Ile cię trzeba cenić, ten tylko się dowie,
Kto cię stracił. Dziś piękność twą w całej ozdobie
Widzę i opisuję, bo tęsknię po tobie. 
Panno święta, co Jasnej bronisz Częstochowy
I w Ostrej świecisz Bramie! Ty, co gród zamkowy
Nowogródzki ochraniasz z jego wiernym ludem!
Jak mnie dziecko do zdrowia powróciłaś cudem
(Gdy od płaczącej matki, pod Twoją opiekę
Ofiarowany, martwą podniosłem powiekę;
I zaraz mogłem pieszo, do Twych świątyń progu
Iść za wrócone życie podziękować Bogu),
Tak nas powrócisz cudem na Ojczyzny łono. 
Tymczasem przenoś moją duszę utęsknioną
Do tych pagórków leśnych, do tych łąk zielonych,
Szeroko nad błękitnym Niemnem rozciągnionych;
Do tych pól malowanych zbożem rozmaitem,
Wyzłacanych pszenicą, posrebrzanych żytem;
Gdzie bursztynowy świerzop, gryka jak śnieg biała,
Gdzie panieńskim rumieńcem dzięcielina pała,
A wszystko przepasane jakby wstęgą, miedzą
Zieloną, na niej z rzadka ciche grusze siedzą. 
Śród takich pól przed laty, nad brzegiem ruczaju,
Na pagórku niewielkim, we brzozowym gaju,
Stał dwór szlachecki, z drzewa, lecz podmurowany;
Świeciły się z daleka pobielane ściany,
Tym bielsze, że odbite od ciemnej zieleni
Topoli, co go bronią od wiatrów jesieni. 
Dom mieszkalny niewielki, lecz zewsząd chędogi,
I stodołę miał wielką, i przy niej trzy stogi
Użątku, co pod strzechą zmieścić się nie może.
Widać, że okolica obfita we zboże,
I widać z liczby kopic, co wzdłuż i wszerz smugów 
Świecą gęsto jak gwiazdy, widać z liczby pługów
Orzących wcześnie łany ogromne ugoru,
Czarnoziemne, zapewne należne do dworu,
Uprawne dobrze na kształt ogrodowych grządek:
Że w tym domu dostatek mieszka i porządek.
Brama na wciąż otwarta przechodniom ogłasza,
Że gościnna, i wszystkich w gościnę zaprasza. 
Właśnie dwukonną bryką wjechał młody panek
I obiegłszy dziedziniec zawrócił przed ganek.
Wysiadł z powozu; konie porzucone same,
Szczypiąc trawę ciągnęły powoli pod bramę.
We dworze pusto: bo drzwi od ganku zamknięto
Zaszczepkami i kołkiem zaszczepki przetknięto.
Podróżny do folwarku nie biegł sług zapytać,
Odemknął, wbiegł do domu, pragnął go powitać.
Dawno domu nie widział, bo w dalekim mieście
Kończył nauki, końca doczekał nareszcie.
Wbiega i okiem chciwie ściany starodawne
Ogląda czule, jako swe znajome dawne.
Też same widzi sprzęty, też same obicia,
Z którymi się zabawiać lubił od powicia,
Lecz mniej wielkie, mniej piękne niż się dawniej zdały.
I też same portrety na ścianach wisiały:
Tu Kościuszko w czamarce krakowskiej, z oczyma
Podniesionymi w niebo, miecz oburącz trzyma;
Takim był, gdy przysięgał na stopniach ołtarzów,
Że tym mieczem wypędzi z Polski trzech mocarzów,
Albo sam na nim padnie. Dalej w polskiej szacie
Siedzi Rejtan, żałośny po wolności stracie;
W ręku trzyma nóż ostrzem zwrócony do łona,
A przed nim leży Fedon i żywot Katona.
Dalej Jasiński, młodzian piękny i posępny;
Obok Korsak, towarzysz jego nieodstępny:
Stoją na szańcach Pragi, na stosach Moskali,
Siekąc wrogów, a Praga już się wkoło pali.
Nawet stary stojący zegar kurantowy 
W drewnianej szafie poznał, u wniścia alkowy;
I z dziecinną radością pociągnął za sznurek,
By stary Dąbrowskiego usłyszeć mazurek. 
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(f"Using device: {device}")

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Translate Polish to English
tokenizer.src_lang = "pl_PL"
max_length = model.config.max_length

# Tokenize the article and split if necessary
encoded_pl = tokenizer(article_pl, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

# Generate translation
generated_tokens = model.generate(
    **encoded_pl,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
)

translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(translated_text)
