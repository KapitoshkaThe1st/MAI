using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;
using Microsoft.Bot.Schema;
using System.Net;
using System.IO;

namespace Microsoft.BotBuilderSamples.Bots
{
    public partial class DietBot : ActivityHandler
    {
        public string Newline(){
            return "\n\n";
        }

        public string Bold(string text){
            return $"**{text}**";
        }

        public string Italic(string text){
            return $"*{text}*";
        }

        private FoodRecognizer foodRecognizer;
        private QnAHandler qnah;
        private BlobStorageHandler bsh;
        
        List<Record> nutrients;

        Dictionary<string, int> userPromptingContext;
        Dictionary<string, PersonalDataRecord> userPromptingRecords;

        private readonly BotState _userState;
        private readonly BotState _conversationState;

        public DietBot(ConversationState conversationState, UserState userState) : base() {
            _conversationState = conversationState;
            _userState = userState;

            userPromptingContext = new Dictionary<string, int>();
            userPromptingRecords = new Dictionary<string, PersonalDataRecord>();

            foodRecognizer = new FoodRecognizer();

            qnah = new QnAHandler();

            bsh = new BlobStorageHandler();
            bsh.Init();

            string localPrefix = Path.GetTempPath();

            string nutrientsCsvBlobPath = "dishesPOG.csv";
            string nutrientsCsvLocalPath = Path.Combine(localPrefix + nutrientsCsvBlobPath);

            bsh.Download(nutrientsCsvBlobPath, nutrientsCsvLocalPath).Wait();

            nutrients = CSVHandler.ReadFromCsv<Record>(nutrientsCsvLocalPath);
        }

        protected async Task SendFormattedMessage(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken, string text){
            IMessageActivity message =  Microsoft.Bot.Schema.Activity.CreateMessageActivity();
            message.Text = text;
            message.TextFormat = "markdown";
            message.Locale = "en-Us";
            await turnContext.SendActivityAsync(message, cancellationToken);
        }

        protected override async Task OnMessageActivityAsync(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken)
        {
            var conversationStateAccessors = _conversationState.CreateProperty<ConversationFlow>(nameof(ConversationFlow));
            var flow = await conversationStateAccessors.GetAsync(turnContext, () => new ConversationFlow(), cancellationToken);

            var userStateAccessors = _userState.CreateProperty<PersonalDataRecord>(nameof(PersonalDataRecord));
            var profile = await userStateAccessors.GetAsync(turnContext, () => new PersonalDataRecord(), cancellationToken);

            IMessageActivity a = turnContext.Activity;

            string replyText = "";

            string userId = a.From.Id;

            bool accountExists = bsh.Exists(Path.Combine(userId, "pers.csv")).Result;
            if(!accountExists){
                if(a.Text == "/start"){
                    string welcomeText = $"Welcome to " + Italic("Diet Bot") + "! First time with us?" + Newline() 
                                + $"Let's create user account for you." + Newline();
                    await turnContext.SendActivityAsync(welcomeText, null, null, cancellationToken);
                }

                await FillOutUserProfileAsync(flow, profile, turnContext, cancellationToken);
            }
            else if(a.Attachments != null && a.Attachments.Count > 0){
                foreach (var item in a.Attachments)
                {
                    var remoteFileUrl = item.ContentUrl;

                    string fileName = userId + DateTime.Now.ToString().Replace(' ', '_').Replace('.', '_').Replace(':', '_');
                    var localFileName = Path.Combine(Path.GetTempPath(), fileName).Replace('-', '_');

                    using (var webClient = new WebClient())
                    {
                        webClient.DownloadFile(remoteFileUrl, localFileName);
                    }

                    var res = await foodRecognizer.MakePredictionRequest(localFileName);

                    replyText += "Your dish is recognized as " + Bold($"{res.tag.ToUpper()}") + "." + Newline();

                    Record curDishRec = null;
                    foreach (var r in nutrients)
                    {
                        if(r.product_name.ToUpper() == res.tag.ToUpper())
                        {
                            curDishRec = r;
                            break;
                        }
                    }

                    replyText += $"It has:" + Newline() + 
                                $"calories: {curDishRec.calories}," + Newline() + 
                                $"protein: {curDishRec.protein}g," + Newline() +  
                                $"fat: {curDishRec.fat}g," + Newline() + 
                                $"carbohydrate: {curDishRec.carbohydrate}g" + Newline();


                    string localPrefix = Path.GetTempPath();

                    string eatingCsvLocal = Path.Combine(localPrefix, userId + "eat.csv");
                    string eatingCsvBlob = Path.Combine(userId, "eat.csv");

                    string personalCsvLocal = Path.Combine(localPrefix, userId + "pers.csv");
                    string personalCsvBlob = Path.Combine(userId, "pers.csv");

                    EatingRecord rec = new EatingRecord{product_name = res.tag, calories = curDishRec.calories,
                                                        protein = curDishRec.protein, carbohydrate = curDishRec.carbohydrate,
                                                        date = DateTime.Now.ToString(), fat = curDishRec.fat};

                    if(!File.Exists(personalCsvLocal)){
                        bsh.Download(personalCsvBlob, personalCsvLocal).Wait();
                    }

                    var pers = CSVHandler.ReadFromCsv<PersonalDataRecord>(personalCsvLocal);
                    PersonalDataRecord personalData = pers[0];

                    if(!File.Exists(eatingCsvLocal)){
                        bsh.Download(eatingCsvBlob, eatingCsvLocal).Wait();
                    }

                    List<EatingRecord> eatings = CSVHandler.ReadFromCsv<EatingRecord>(eatingCsvLocal);

                    float todayCalories = curDishRec.calories;
                    float todayProtein = curDishRec.protein;
                    float todayFat = curDishRec.fat;
                    float todayCarbohydrate = curDishRec.carbohydrate;

                    string piePlotUrl = $"http://84.201.131.5/test/pie/{todayProtein.ToString().Replace(',', '.')},{todayFat.ToString().Replace(',', '.')},{todayCarbohydrate.ToString().Replace(',', '.')}";

                    replyText += "Nutrient ratio:" + Newline() + piePlotUrl + Newline();

                    int n = eatings.Count;
                    
                    for(int i = n-1; i >= 0; --i){
                        if(Convert.ToDateTime(eatings[i].date).Date < DateTime.Now.Date)
                            break;                        
                        todayCalories += eatings[i].calories;
                        todayProtein += eatings[i].protein;
                        todayFat += eatings[i].fat;
                        todayCarbohydrate += eatings[i].carbohydrate;
                    }

                    replyText += "Bon appetit!" + Newline();
                    replyText += $"Daily nutrient norm:" + Newline() +
                    Italic("calories") + $": {todayCalories} \\ {personalData.caloriesNorm}" + Newline() +
                    Italic("protein") + $": {todayProtein} \\ {personalData.proteinNorm}" + Newline() +
                    Italic("fat") + $": {todayFat} \\ {personalData.fatNorm}" + Newline() + 
                    Italic("carbohydrate") + $": {todayCarbohydrate} \\ {personalData.carbohydrateNorm}"+ Newline();

                    string dayPfcUrl = $"http://84.201.131.5/test/bar/{todayProtein.ToString().Replace(',', '.')},{todayFat.ToString().Replace(',', '.')},{todayCarbohydrate.ToString().Replace(',', '.')},{personalData.proteinNorm.ToString().Replace(',', '.')},{personalData.fatNorm.ToString().Replace(',', '.')},{personalData.carbohydrateNorm.ToString().Replace(',', '.')}";
                    string dayCalUrl = $"http://84.201.131.5/test/calories/{todayCalories.ToString().Replace(',', '.')},{personalData.caloriesNorm.ToString().Replace(',', '.')}";

                    replyText += Newline() + Newline() + "Todays calories: " + Newline() + dayCalUrl + Newline() + "Todays nutrients:" + Newline() + dayPfcUrl;

                    eatings.Add(rec);
                    CSVHandler.WriteToCsv(eatings, eatingCsvLocal);
                    bsh.Upload(eatingCsvLocal, eatingCsvBlob).Wait();
                }
            }
            else{
                string userName = a.From.Name;

                string mode = qnah.GenerateAnswer(a.Text).Result;

                if(mode.Contains("stats"))
                {
                    string localPrefix = Path.GetTempPath();

                    string eatingCsvLocal = Path.Combine(localPrefix, userId + "eat.csv");
                    string eatingCsvBlob = Path.Combine(userId, "eat.csv");

                    string personalCsvLocal = Path.Combine(localPrefix, userId + "pers.csv");
                    string personalCsvBlob = Path.Combine(userId, "pers.csv");

                    if(!File.Exists(personalCsvLocal)){
                        bsh.Download(personalCsvBlob, personalCsvLocal).Wait();
                    }

                    var pers = CSVHandler.ReadFromCsv<PersonalDataRecord>(personalCsvLocal);
                    PersonalDataRecord personalData = pers[0];

                    if(!File.Exists(eatingCsvLocal)){
                        bsh.Download(eatingCsvBlob, eatingCsvLocal).Wait();
                    }

                    List<EatingRecord> eatings = CSVHandler.ReadFromCsv<EatingRecord>(eatingCsvLocal);

                    int n = eatings.Count;
                    DateTime curDate = DateTime.Now.Date;
                    string curDateString = $"{curDate.Year},{curDate.Month},{curDate.Day}";
                    
                    if(mode.Contains("week")){
                        float[] dayCalories = new float[7];
                        float[] dayProtein = new float[7];
                        float[] dayFat = new float[7];
                        float[] dayCarbohydrate = new float[7];

                        int j = n-1; 
                        for(int i = 6; i >= 0; --i){
                            for(;j >= 0; --j){
                                if(Convert.ToDateTime(eatings[j].date).Date < curDate)
                                    break;
                                dayCalories[i] += eatings[j].calories;
                                dayProtein[i] += eatings[j].protein;
                                dayFat[i] += eatings[j].fat;
                                dayCarbohydrate[i] += eatings[j].carbohydrate;
                            }
                            curDate = curDate.AddDays(-1);
                        }

                        string weekPfcUrl = "http://84.201.131.5/test/linechart/";

                        for(int i = 0; i < 7; ++i){
                            weekPfcUrl += $"{dayProtein[i].ToString().Replace(',', '.')},";
                        }
                        for(int i = 0; i < 7; ++i){
                            weekPfcUrl += $"{dayFat[i].ToString().Replace(',', '.')},";
                        }
                        for(int i = 0; i < 7; ++i){
                            weekPfcUrl += $"{dayCarbohydrate[i].ToString().Replace(',', '.')},";
                        }

                        weekPfcUrl += @$"{personalData.proteinNorm.ToString().Replace(',', '.')},"
                            + $"{personalData.fatNorm.ToString().Replace(',', '.')},"
                            + $"{personalData.carbohydrateNorm.ToString().Replace(',', '.')},";

                        weekPfcUrl += curDateString;

                        replyText += "Week statistic: " + Newline() + "weekly nutrients: " + Newline() + weekPfcUrl;

                    }
                    else if(mode.Contains("day"))
                    {
                        float dayCalories = 0.0f;
                        float dayProtein = 0.0f;
                        float dayFat = 0.0f;
                        float dayCarbohydrate = 0.0f;

                        for(int j = n-1; j >= 0; --j){
                            if(Convert.ToDateTime(eatings[j].date).Date < curDate)
                                break;
                            dayCalories += eatings[j].calories;
                            dayProtein += eatings[j].protein;
                            dayFat += eatings[j].fat;
                            dayCarbohydrate += eatings[j].carbohydrate;
                        }

                        string dayPfcUrl = $"http://84.201.131.5/test/bar/{dayProtein.ToString().Replace(',', '.')},{dayFat.ToString().Replace(',', '.')},{dayCarbohydrate.ToString().Replace(',', '.')},{personalData.proteinNorm.ToString().Replace(',', '.')},{personalData.fatNorm.ToString().Replace(',', '.')},{personalData.carbohydrateNorm.ToString().Replace(',', '.')}";
                        string dayCalUrl = $"http://84.201.131.5/test/calories/{dayCalories.ToString().Replace(',', '.')},{personalData.caloriesNorm.ToString().Replace(',', '.')}";

                        replyText += "Day statistic: " + Newline() + "daily nutrients: " + Newline() + dayPfcUrl + Newline() + "daily calories: " + Newline() + dayCalUrl;
                    }
                }
                else{
                    replyText += mode;
                }
            }

            await SendFormattedMessage(turnContext, cancellationToken, replyText);

            await _conversationState.SaveChangesAsync(turnContext, false, cancellationToken);
            await _userState.SaveChangesAsync(turnContext, false, cancellationToken);
        }

        protected override async Task OnMembersAddedAsync(IList<ChannelAccount> membersAdded, ITurnContext<IConversationUpdateActivity> turnContext, CancellationToken cancellationToken)
        {
            var welcomeText = "Welcome to DietBot!";

            foreach (var member in membersAdded)
            {
                if (member.Id != turnContext.Activity.Recipient.Id)
                {
                    await turnContext.SendActivityAsync(MessageFactory.Text(welcomeText, welcomeText), cancellationToken);
                    await turnContext.SendActivityAsync($"Hi there - {member.Name}. Glad to see you!", cancellationToken: cancellationToken);
                }
            }
        }
    }
}
