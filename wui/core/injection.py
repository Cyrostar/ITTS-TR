import random

def tr_corpus(freq: int = 1, shuffle: bool = False) -> str:
    
    if not isinstance(freq, int) or freq < 1:
        raise ValueError("freq must be a positive integer")
        
    en = [
        "box", "equal", "exact", "example", "excited", "excuse", "exercise", "exit", 
        "expect", "expensive", "expert", "explain", "extra", "extreme", "fix", "fixed", 
        "liquid", "mix", "mixed", "mixing", "next", "paperwork", "quality", "quarter", 
        "queen", "quest", "question", "quick", "quicken", "quickly", "quiet", "quieten", 
        "quit", "quitters", "quite", "relax", "relaxer", "six", "squash", "square", 
        "squeeze", "tax", "taxi", "wait", "wake", "walk", "walkway", "want", "warm", 
        "wash", "watch", "water", "wax", "waxed", "waxwing", "way", "weak", "weather", 
        "week", "welcome", "white", "wide", "wife", "win", "wind", "window", "wing", 
        "winter", "wish", "with", "woman", "wood", "word", "work", "worksheet", 
        "world", "write", "wrong"
    ]
    
    tr = [
        "çağdaş", "bağışçı", "aşçı", "değiş", "bağış", "çarşı", "kaçış", "ağaç",
        "çay", "çocuk", "çok", "ağır", "sağ", "öğrenci", "şimdi", "şeker",
        "akşam", "eşlik", "buluşma", "değişim", "çiçek", "güneş", "yağmur", "düş",
        "çalış", "doğru", "ışık", "kuş", "dağ", "kağıt", "soğuk", "sıcak", "çabuk",
        "şans", "şarkı", "bağ", "çağrı", "çizgi", "düğme", "eşya", "genç", "giriş",
        "görüş", "gülüş", "gümüş", "hafif", "içerisi", "işaret", "kaşık", "koşmak",
        "küçük", "maaş", "mesaj", "öğle", "öğretmen", "pişman", "sağlık", "seçim",
        "sevgi", "sırdaş", "şaka", "şehir", "şoför", "şüphe", "uçak",
        "uğraş", "yavaş", "yoğurt", "yüzleş", "şişe"
    ]
    
    vowels = [
        "câriye", "câmi", "câri", "câhil", "kâğıt", "kâr", "kâtip", "hâl", "hâlâ", "hâkim",
        "şâir", "şâhit", "şehâdet", "târih", "mâna", "mâkul", "nâdir", "nâzik", "sâkin",
        "bîhaber", "bîçare", "resmî", "millî", "ilmî", "hakîkat",
        "sükût", "sükûnet", "sûret", "usûl"
    ]
    
    spoken = [
        "noluyo", "noldu", "nolcak", "noluyosun", "nolmuş", "noluyodu", "noluyo ya", "burda mısın", 
        "burda bekle", "burda kal", "burda ne var", "burda çalışıyo", "burda oturuyo", "bi gün", 
        "bi şey", "bi dakka", "bi tane", "bi kere", "bi bak", "bi ara", "bi problem", "bi insan", 
        "bi konu", "geliyo", "gidiyo", "bakıyo", "yazıyo", "konuşuyo", "bekliyo", "çalışıyo", 
        "düşünüyo", "anlıyo", "istiyo", "yapmıyo", "sevmiyo", "yapıyo", "ediyo", "oluyo", 
        "kalıyo", "duruyo", "görüyo", "biliyo", "sanıyo", "söylüyo", "diyo", "seviyo", 
        "etmiyo", "vermiyo", "alıyo", "tutuyo", "bırakıyo", "açıyo", "kapatıyo", 
        "giriyo", "çıkıyo", "oturuyo", "kalkıyo", "yatıyo", "uyuyo", "koşuyo", 
        "yürüyüyo", "dönüyo", "arıyo", "buluyo", "kaybediyo", "seçiyo", "deniyo", 
        "başlıyo", "bitiyo", "taşıyo", "atıyo", "çekiyo", "itiyo", "soruyo", 
        "cevaplıyo", "dinliyo", "anlatıyo", "hatırlıyo", "unutuyo", "öğreniyo", 
        "öğretiyo", "kazanıyo", "kullanıyo", "denetliyo", "kontrol ediyo", 
        "planlıyo", "hazırlanıyo", "izliyo", "seyrediyo", "okuyuo", "çiziyo", 
        "paylaşıyo", "gönderiyo", "getiriyo", "götürüyo", "değişiyo", "gelişiyo", 
        "artıyo", "azalıyo", "büyüyo", "küçülüyo", "kapanıyo", "yaşıyo", "ölüyo"
    ]
    
    months = [
        "ocak", "şubat", "mart", 
        "nisan", "mayıs", "haziran", 
        "temmuz", "ağustos", "eylül", 
        "ekim", "kasım", "aralık"
    ]
    
    seasons = ["ilkbahar", "yaz", "sonbahar", "kış"]
    
    days = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"]

    zodiac = [
        "koç", "boğa", "ikizler", 
        "yengeç", "aslan", "başak", 
        "terazi", "akrep", "yay", 
        "oğlak", "kova", "balık"
    ]
    
    planets = ["merkür", "venüs", "dünya", "mars", "jüpiter", "satürn", "uranüs", "neptün", "plüton"]
    
    directions = [
        "sağ", "sol", "yukarı", "aşağı",
        "ileri", "geri", "doğu", "batı",
        "kuzey", "güney"
    ]
    
    rainbow = ["kırmızı", "turuncu", "sarı", "yeşil", "mavi", "lacivert", "mor"]
    
    daytimes = ["sabah", "öğle", "ikindi", "akşam", "gece", "yatsı"]
    
    notes = ["do", "re", "mi", "fa", "sol", "la", "si"]
    
    numbers = [
        "bir", "iki", "üç", "dört", "beş",
        "altı", "yedi", "sekiz", "dokuz", "on",
        "onbir", "oniki", "onüç", "ondört", "onbeş",
        "onaltı", "onyedi", "onsekiz", "ondokuz", "yirmi",
        "yirmibir", "yirmiiki", "yirmiüç", "yirmidört", "yirmibeş",
        "yirmialtı", "yirmiyedi", "yirmisekiz", "yirmidokuz", "otuz",
        "otuzbir", "otuziki", "otuzüç", "otuzdört", "otuzbeş",
        "otuzaltı", "otuzyedi", "otuzsekiz", "otuzdokuz", "kırk",
        "kırkbir", "kırkiki", "kırküç", "kırkdört", "kırkbeş",
        "kırkaltı", "kırkyedi", "kırksekiz", "kırkdokuz", "elli",
        "ellibir", "elliiki", "elliüç", "ellidört", "ellibeş",
        "ellialtı", "elliyedi", "elliseki̇z", "ellidokuz", "altmış",
        "altmışbir", "altmışiki", "altmışüç", "altmışdört", "altmışbeş",
        "altmışaltı", "altmışyedi", "altmışsekiz", "altmışdokuz", "yetmiş",
        "yetmişbir", "yetmişiki", "yetmişüç", "yetmişdört", "yetmişbeş",
        "yetmişaltı", "yetmişyedi", "yetmişsekiz", "yetmişdokuz", "seksen",
        "seksenbir", "sekseniki", "seksenüç", "seksendört", "seksenbeş",
        "seksenaltı", "seksen yedi", "seksensekiz", "seksendokuz", "doksan",
        "doksanbir", "doksaniki", "doksanüç", "doksandört", "doksanbeş",
        "doksanaltı", "doksanyedi", "doksansekiz", "doksandokuz", "yüz"
    ]
    
    vegetables = [
        "patates", "domates", "soğan", "sarımsak", "havuç",
        "patlıcan", "kabak", "salatalık", "biber", "marul",
        "lahana", "karnabahar", "brokoli", "pırasa", "ıspanak",
        "pazı", "roka", "tere", "fasulye", "bezelye",
        "bakla", "nohut", "mercimek", "mısır", "kereviz",
        "turp", "pancar", "enginar", "bamya", "kuşkonmaz",
        "rezene", "dereotu", "maydanoz", "fesleğen", "nane",
        "kekik", "adaçayı", "kapya biber", "sivri biber", "dolmalık biber",
        "acı biber", "kırmızı biber", "yeşil biber", "çarliston biber", "mantar",
        "avokado", "zencefil", "zerdeçal", "yer elması", "taze soğan",
        "arpacık soğan", "brüksel lahanası", "kırmızı lahana", "mor lahana", "kara lahana",
        "çin lahanası", "iceberg marul", "kıvırcık marul", "hardal otu", "hindiba",
        "semizotu", "şalgam", "tatlı patates", "mor patates", "edamame",
        "soya fasulyesi", "börülce", "barbunya", "taze fasulye", "kabak çiçeği",
        "bal kabağı", "acur", "turşuluk salatalık", "jalapeno", "habanero",
        "dolma biber", "köz biber", "mini mısır", "frenk soğanı", "yeşil soğan",
        "sarımsak filizi", "kişniş", "roka çiçeği", "su teresi", "alabaş",
        "kolrabi", "radika", "deniz börülcesi", "madımak", "ebegümeci",
        "ısırgan", "labada", "hindiba kökü", "kudret narı", "tatlı kabak",
        "kabak çekirdeği", "pancar kökü", "turp otu", "karalahana", "kereviz sapı",
        "kereviz kökü", "mini kabak", "spagetti kabağı", "butternut kabağı", "kabocha",
        "romanesco", "karnıyarık otu", "çörek otu yaprağı", "hardal yaprağı", "pak choi",
        "bok choy", "tat soi", "mizuna", "wasabi kökü", "lotus kökü",
        "manyok", "taro", "okra", "chili biber", "acı süs biberi"
    ]
    
    fruits = [
        "elma", "armut", "muz", "portakal", "mandalina",
        "limon", "greyfurt", "nar", "üzüm", "çilek",
        "kiraz", "vişne", "kayısı", "şeftali", "erik",
        "nektarin", "incir", "hurma", "ananas", "mango",
        "kivi", "papaya", "avokado", "karpuz", "kavun",
        "ayva", "narenciye", "frambuaz", "böğürtlen", "yaban mersini",
        "ahududu", "dut", "muz", "çarkıfelek meyvesi", "liçi",
        "rambutan", "longan", "ejder meyvesi", "guava", "jackfruit",
        "durian", "yıldız meyvesi", "kızılcık", "muşmula", "alıç",
        "keçiboynuzu", "kuşburnu", "hünnap", "bergamot", "pomelo",
        "kan portakalı", "tangelo", "kumkuat", "yuzu", "bergamot limonu",
        "turna yemişi", "goji berry", "aronya", "altın çilek", "pitanga",
        "sapodilla", "mangosteen", "salak", "feijoa", "breadfruit",
        "soursop", "tamarillo", "tamarind", "bael", "ackee",
        "bilberry", "blackberry", "cranberry", "currant", "elderberry",
        "mirabelle", "reineclaude", "jabuticaba", "cherimoya", "cempedak",
        "marang", "medlar", "persimmon", "plantain", "ugli fruit"
    ]
    
    trees = [
        "meşe", "çam", "ladin", "köknar", "sedir",
        "ardıç", "servi", "kavak", "söğüt", "huş",
        "kayın", "gürgen", "kestane", "ceviz", "fındık",
        "badem", "zeytin", "defne", "çınar", "akçaağaç",
        "dişbudak", "karaağaç", "ıhlamur", "akasya", "erguvan",
        "incir", "dut", "nar", "elma", "armut",
        "kiraz", "vişne", "kayısı", "şeftali", "erik",
        "ayva", "limon", "portakal", "mandalina", "turunç",
        "greyfurt", "hurma", "palmiye", "muz", "avokado",
        "mango", "kakao", "kauçuk", "tik", "okaliptüs",
        "sandal ağacı", "baobab", "sekoya", "selvi", "mazı",
        "sumak", "keçiboynuzu", "alıç", "muşmula", "hünnap",
        "kızılcık", "çitlembik", "sakız ağacı", "manolya", "katalpa",
        "mimoza", "dişbudak akçaağacı", "at kestanesi", "kızıl meşe", "karaçam",
        "sarıçam", "toros sediri", "liban sediri", "mavi ladin", "kanada ladini",
        "titrek kavak", "kara kavak", "beyaz söğüt", "salkım söğüt", "amerikan ceviz",
        "japon akçaağacı", "kırmızı akçaağaç", "şeker akçaağacı", "amerikan dişbudak", "kafkas ıhlamuru",
        "erik ağacı", "nektarin ağacı", "bergamot ağacı", "pomelo ağacı", "yuzu ağacı",
        "tangelo ağacı", "kumkuat ağacı", "narenciye ağacı", "trabzon hurması", "guava ağacı",
        "jackfruit ağacı", "durian ağacı", "liçi ağacı", "longan ağacı", "rambutan ağacı",
        "tamarind ağacı", "bael ağacı", "breadfruit ağacı", "soursop ağacı", "cherimoya ağacı",
        "jabuticaba ağacı", "cempedak ağacı", "marang ağacı", "ugli fruit ağacı", "persimmon ağacı"
    ]
    
    flowers = [
        "gül", "lale", "sümbül", "nergis", "zambak",
        "orkide", "menekşe", "papatya", "karanfil", "begonya",
        "sardunya", "petunya", "frezya", "şakayık", "nilüfer",
        "lotus", "kasımpatı", "kamelya", "gardenya", "leylak",
        "yasemin", "mimoza", "ortanca", "çuha çiçeği", "lavanta",
        "kardelen", "çiğdem", "erguvan", "hanımeli", "fulya",
        "akasya çiçeği", "yıldız çiçeği", "kadife çiçeği", "aslanağzı", "cam güzeli",
        "ortanca çiçeği", "ortanca", "akşam sefası", "ateş çiçeği", "turna gagası",
        "gelincik", "anemon", "iris", "süsen", "çan çiçeği",
        "mor salkım", "begonvil", "manolya çiçeği", "zakkum", "reyhan çiçeği",
        "mine çiçeği", "gazanya", "hibiskus", "ortensia", "kına çiçeği",
        "kral tacı", "sıklamen", "kır çiçeği", "ayçiçeği", "kamkat çiçeği",
        "badem çiçeği", "erik çiçeği", "kiraz çiçeği", "şeftali çiçeği", "nar çiçeği",
        "iğde çiçeği", "okside papatya", "afrika menekşesi", "kalanşo", "antoryum",
        "glayöl", "amarilis", "dahlia", "lisianthus", "ranunkulus",
        "asters", "ortaköy zambağı", "tutya çiçeği", "verbena", "delphinium",
        "petek çiçeği", "sarmaşık gülü", "yaban gülü", "dağ lalesi", "kum zambağı",
        "maviş çiçeği", "acı çiğdem", "dağ sümbülü", "kış gülü", "ipek çiçeği"
    ]
    
    organs = [
        "beyin", "kalp", "akciğer", "karaciğer", "böbrek",
        "mide", "ince bağırsak", "kalın bağırsak", "pankreas", "dalak",
        "safra kesesi", "yemek borusu", "soluk borusu", "tiroit", "paratiroit",
        "hipofiz", "epifiz", "böbreküstü bezleri", "mesane", "idrar yolu",
        "üreter", "üretra", "prostat", "rahim", "yumurtalık",
        "testis", "vajina", "penis", "plasenta", "deri",
        "göz", "kulak", "burun", "dil", "ağız",
        "dişler", "bademcikler", "lenf düğümleri", "kemik iliği", "kaslar",
        "sinirler", "omurilik", "hipotalamus", "beyincik", "korteks",
        "aort", "atardamar", "toplardamar", "kılcal damar", "lenf damarları",
        "timus", "bronş", "bronşçuk", "alveol", "diyafram",
        "ince bağırsak villusları", "rektum", "anüs", "safra kanalı", "pankreas kanalı",
        "idrar kesesi", "epididimis", "vas deferens", "serviks", "fallop tüpleri",
        "endometrium", "miyometrium", "perikard", "plevra", "periton",
        "retina", "kornea", "iris", "pupilla", "optik sinir",
        "kulak zarı", "orta kulak", "iç kulak", "koklea", "östaki borusu",
        "tükürük bezleri", "karaciğer lobu", "dalak kapsülü", "bağırsak mezenteri", "lenf bezi"
    ]
    
    mammals = [
        "kedi", "köpek", "at", "inek", "koyun",
        "keçi", "eşek", "deve", "fil", "aslan",
        "kaplan", "leopar", "çita", "puma", "jaguar",
        "vaşak", "ayı", "kurt", "tilki", "çakal",
        "sırtlan", "porsuk", "rakun", "panda", "kırmızı panda",
        "koala", "kanguru", "wombat", "keseli sıçan", "opossum",
        "zebra", "zürafa", "gergedan", "su aygırı", "antilop",
        "ceylan", "geyik", "sığın", "ren geyiği", "bizon",
        "manda", "bufalo", "yaban öküzü", "domuz", "yaban domuzu",
        "aygır", "katır", "lama", "alpaka", "yak",
        "sincap", "uçan sincap", "tavşan", "tavşancık", "kobay",
        "hamster", "fare", "sıçan", "köstebek", "kirpi",
        "gelincik", "sansar", "samur", "vizon", "su samuru",
        "fok", "mors", "deniz aslanı", "balina", "yunus",
        "orka", "narval", "ispermeçet", "beluga", "grönland balinası",
        "şempanze", "goril", "orangutan", "babun", "makak",
        "kapuçin", "marmoset", "lemur", "maki", "tarsiyer",
        "maymun", "insan", "tapir", "karakulak", "kar leoparı",
        "pangolin", "armadillo", "tembel hayvan", "mirket", "dikdik",
        "okapi", "fossa", "ayı köpeği", "korsak tilkisi", "manul",
        "dağ keçisi", "dağ koyunu", "misk öküzü", "kunduz", "kapibara",
        "aguti", "vizcacha", "şinşilla", "tenrek", "kirpi faresi"
    ]
    
    birds = [
        "serçe", "güvercin", "martı", "karga", "saksağan",
        "bıldırcın", "keklik", "sülün", "tavus kuşu", "tavuk",
        "horoz", "hindi", "ördek", "kaz", "kuğu",
        "turna", "leylek", "flamingo", "pelikan", "karabatak",
        "albatros", "penguen", "devekuşu", "emu", "nandu",
        "kartal", "şahin", "doğan", "akbaba", "atmaca",
        "baykuş", "kukuma", "toy kuşu", "ibibik", "yusufçuk kuşu",
        "arı kuşu", "saka", "ispinoz", "kanarya", "muhabbet kuşu",
        "papağan", "kakadu", "loriket", "amazon papağanı", "afrika gri papağanı",
        "bülbül", "çalıkuşu", "kızılgerdan", "kırlangıç", "ebabil",
        "ağaçkakan", "yalıçapkını", "turna balıkçıl", "balıkçıl", "kaşıkçı kuşu",
        "ak pelikan", "kara leylek", "ak leylek", "sakarmeke", "su tavuğu",
        "çulluk", "kız kuşu", "martı balıkçıl", "sumru", "fregat kuşu",
        "kar kazı", "angut", "elmabaş patka", "tepeli patka", "deniz ördeği",
        "atmaca kartal", "kızıl şahin", "bozkır kartalı", "şah kartal", "kerkenez",
        "toygar", "bayağı serçe", "dağ bülbülü", "saz delicesi", "çayır kuşu",
        "ak kuyruklu kartal", "balık kartalı", "kızıl akbaba", "kara akbaba", "küçük kerkenez",
        "büyük toy", "küçük toy", "tepeli karabatak", "fiyu", "kılkuyruk",
        "turna kuşu", "kumru", "yeşil papağan", "gökçe güvercin", "altın kartal"
    ]
    
    fish = [
        "hamsi", "istavrit", "çinekop", "lüfer", "levrek",
        "çipura", "palamut", "uskumru", "orfoz", "lagos",
        "mercan", "kefal", "mezgit", "morina", "ringa",
        "dil balığı", "kalkan", "vatoz", "köpekbalığı", "manta vatozu",
        "kılıçbalığı", "orfoz", "trança", "fangri", "minekop",
        "barbunya", "tekir", "iskorpit", "lapin", "sinarit",
        "akya", "sarpa", "kupes", "izmarit", "karagöz",
        "mırmır", "sardalya", "kolyoz", "tombik", "torik",
        "zargana", "uskumru", "tirsi", "çaça", "gümüş balığı",
        "yayın balığı", "sazan", "alabalık", "turna balığı", "tatlı su levreği",
        "tatlı su kefali", "çapak", "kadife balığı", "kızılkanat", "gambusya",
        "japon balığı", "lepistes", "plati", "moli", "beta balığı",
        "zebra balığı", "discus", "ciklet", "melek balığı", "tetra",
        "piranha", "tilapia", "somon", "alaska somonu", "uskumru torik",
        "kefal yavrusu", "çupra", "kefal balığı", "orata", "deniz levreği",
        "mavi yüzgeçli orkinos", "orkinos", "uskumru kolyoz", "kaya balığı", "kırlangıç",
        "trakonya", "iskarmoz", "gelincik balığı", "horozbina", "lipsoz",
        "çitar", "hanos", "tilki balığı", "kum köpekbalığı", "çekiç başlı köpekbalığı",
        "beyaz köpekbalığı", "camgöz", "kefal yavrusu", "kırlangıç balığı", "kum balığı"
    ]
    
    common = [
        "ben", "sen", "o", "biz", "siz",
        "onlar", "bu", "şu", "ne", "kim",
        "nerede", "nasıl", "neden", "niye", "hangi",
        "kaç", "var", "yok", "evet", "hayır",
        "tamam", "peki", "yani", "işte", "falan",
        "filan", "şimdi", "bugün", "yarın", "dün",
        "sonra", "önce", "hep", "hiç", "bazen",
        "belki", "yine", "artık", "daha", "çok",
        "az", "biraz", "şey", "biri", "herkes",
        "kimse", "başka", "aynı", "farklı", "gibi",
        "kadar", "çünkü", "ama", "ve", "ya",
        "ile", "için", "gibi", "kadar", "sonra",
        "önce", "sonra", "çok", "az", "iyi",
        "kötü", "güzel", "çirkin", "büyük", "küçük",
        "uzun", "kısa", "eski", "yeni", "doğru",
        "yanlış", "kolay", "zor", "mümkün", "lazım",
        "gerek", "var", "yok", "olabilir", "değil",
        "mi", "mı", "mu", "mü", "acaba",
        "galiba", "sanırım", "bence", "bana", "sana",
        "ona", "bize", "size", "onlara", "benim",
        "senin", "onun", "bizim", "sizin", "onların",
        "kendim", "kendin", "kendi", "kendiyle", "birlikte",
        "yalnız", "beraber", "hemen", "şimdi", "sonra",
        "az önce", "birazdan", "burada", "orada", "şurada",
        "gel", "git", "geldi", "gitti", "geliyor",
        "gidiyor", "yap", "yaptım", "yapıyorum", "yapacak",
        "et", "etti", "ediyor", "olacak", "oldu",
        "oluyor", "olamaz", "olmak", "ister", "istemek",
        "sev", "sevmek", "seviyorum", "sevdim", "sevecek",
        "nefret", "korku", "korkmak", "bil", "bilmek",
        "biliyorum", "biliyorsun", "biliyor", "bilmiyorum", "gör",
        "görmek", "gördüm", "görüyorum", "bak", "bakmak",
        "duymak", "hissetmek", "düşünmek", "sanmak", "saymak",
        "vermek", "almak", "kalmak", "durmak", "başlamak",
        "bitirmek", "söylemek", "demek", "konuşmak", "susmak",
        "sormak", "cevaplamak", "dinlemek", "anlatmak", "anlamak",
        "anlamamak", "çalışmak", "yapmak", "etmek", "olmak",
        "gitmek", "gelmek", "koymak", "çıkmak", "girmek",
        "açmak", "kapatmak", "tutmak", "bırakmak", "taşımak",
        "atmak", "çekmek", "itmek", "beklemek", "aramak",
        "bulmak", "kaybetmek", "seçmek", "karar vermek", "denemek",
        "başarmak", "başaramamak", "yardım etmek", "istemek", "zorunda",
        "mecbur", "gerek", "lazım", "yeter", "fazla",
        "eksik", "aynen", "tamam", "oldu", "olur",
        "olmaz", "belki", "kesin", "kesinlikle", "asla",
        "her zaman", "bazen", "nadiren", "sık sık", "genelde",
        "şimdi", "hemen", "az sonra", "erken", "geç",
        "bekle", "dur", "haydi", "hadi", "of",
        "eyvallah", "sağ ol", "teşekkürler", "rica ederim", "pardon",
        "lütfen", "buyur", "hoş geldin", "güle güle", "görüşürüz",
        "görüşmek üzere", "kendine iyi bak", "iyi günler", "iyi akşamlar", "iyi geceler",
        "günaydın", "selam", "merhaba", "naber", "ne haber"
    ]
    
    emotions = [
        "mutluluk","sevinç","neşe","haz","keyif",
        "üzüntü","keder","elem","acı",
        "korku","endişe","kaygı","panik","dehşet",
        "öfke","kızgınlık","hiddet","sinir",
        "şaşkınlık","hayret",
        "tiksinti","iğrenme",
        "utanç","mahcubiyet","suçluluk","vicdan azabı",
        "gurur","onur","kibir",
        "kıskançlık","haset",
        "hayranlık","şefkat","merhamet","empati",
        "sevgi","aşk","bağlılık","sadakat",
        "güven","güvensizlik",
        "kırgınlık","affedicilik",
        "terk edilme korkusu","yakınlık","mesafe",
        "umut","umutsuzluk",
        "pişmanlık","özlem","nostalji",
        "merak","heyecan",
        "rahatlama","tatmin","doyum",
        "huzur","sükûnet",
        "coşku","vecd",
        "hayal kırıklığı","sıkıntı","melankoli","yas",
        "tedirginlik","rahatsızlık",
        "boşluk","anlamsızlık","yabancılaşma","varoluşsal kaygı",
        "kabulleniş","içsel çatışma","farkındalık","aydınlanma hissi","teslimiyet",
        "ambivalans","empatik acı","tatlı hüzün","suç ortağı olma hissi",
        "açıklanamaz huzur","açıklanamaz huzursuzluk",
        "rahat","gergin","huzursuz","ferah",
        "yorgun","bitkin","enerjik","uyușuk",
        "canlı","donuk","tükenmiş","motive","isteksiz"
    ]
    
    adjectives = [
        "iyi","kötü","güzel","çirkin","doğru","yanlış",
        "büyük","küçük","uzun","kısa","geniş","dar",
        "yüksek","alçak","kalın","ince","derin","sığ",
        "ağır","hafif","sert","yumuşak",
        "hızlı","yavaş","erken","geç","ani","sürekli",
        "geçici","kalıcı","eski","yeni","modern","klasik",
        "çok","az","fazla","eksik","tam","yarım",
        "yoğun","seyrek","bol","kıt","net","belirgin",
        "sıcak","soğuk","ılık","serin","temiz","kirli",
        "açık","kapalı","parlak","mat","kuru","ıslak",
        "zor","kolay","mümkün","imkansız","gerekli",
        "lazım","yeterli","yetersiz","uygun","uygunsuz",
        "önemli","önemsiz","değerli","değersiz",
        "resmi","gayriresmi","ciddi","basit",
        "zorunlu","istekli","isteksiz",
        "başarılı","başarısız","aktif","pasif",
        "canlı","donuk","güçlü","zayıf",
        "hızlı","yavaş","verimli","etkili",
        "aynı","farklı","benzer","uyumlu","uyumsuz"
    ]
    
    slang = [
        "lan","ulan","la","kanka","aga","abi","bro","birader","moruk",
        "hacı","reis","dayı","usta","oha","of","vay","ayyy","ya","yaaa",
        "hmm","hah","aynen","aynen öyle","ok","eyvallah","sağol","yok ya",
        "nah","he he","hıh","boşver","salla geç","boş yap","ne alaka",
        "ne diyosun","ne diyim","hadi ya","hadi lan","bi dur","sakin ol",
    
        "trip atmak","triplenmek","tribe girme","kafaya takma","kafayı yedim",
        "kafam almıyor","kafam kaldırmıyor","kafam dolu","kafam kazan gibi",
        "modum yok","canım sıkkın","canım istemiyor","moral sıfır",
        "keyfim kaçtı","sinirim bozuk", "gostlamak",
    
        "takılmak","kopmak","dağılmak","çakmak","sallamak",
        "geçiştirmek","tınlamamak","umursamamak","sal gitsin",
        "kafana takma","kafayı yeme",
    
        "acayip","fena","baya","aşırı","efsane","manyak",
        "süper","müthiş","felaket","berbat","rezalet",
        "bomba","sağlam","harbi","cidden","vallahi","billahi",

        "salak","aptal","gerizekalı","dangalak","mal","öküz",
        "hödük","beyinsiz","embesil","andaval",
        "yavşak","şerefsiz","namussuz","haysiyetsiz","pislik",
        "piç","pezevenk","orospu","gavat","puşt",

        "hasiktir","siktir","siktir git","siktir lan"
    ]
    
    words = (
        en + tr + vowels + spoken + months + seasons + days + zodiac + 
        planets + directions + rainbow + daytimes + notes + numbers +
        vegetables + fruits + trees + flowers + organs + mammals + 
        birds + fish + common + emotions + adjectives + slang
    )
        
    corpus = sorted(list(set(words)))
    
    final = corpus * freq
    
    if shuffle:
        random.shuffle(final)

    text = " ".join(final)
    
    return text