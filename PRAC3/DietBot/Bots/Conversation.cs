using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;

using System.IO;

using System.Globalization;

namespace Microsoft.BotBuilderSamples.Bots
{
    public class ConversationFlow
    {
        public enum Question
        {
            None,
            Height,
            Weight,
            Age,
            Sex,
            PhysActCoef,
            Finished
        }

        public Question LastQuestionAsked { get; set; } = Question.None;
    }

    public partial class DietBot : ActivityHandler
    {
        private async Task FillOutUserProfileAsync(ConversationFlow flow, PersonalDataRecord profile, ITurnContext turnContext, CancellationToken cancellationToken)
        {
            var input = turnContext.Activity.Text?.Trim();
            string message;

            switch (flow.LastQuestionAsked)
            {
                case ConversationFlow.Question.None:
                    await turnContext.SendActivityAsync("Let's get started. How tall are you?", null, null, cancellationToken);
                    flow.LastQuestionAsked = ConversationFlow.Question.Height;
                    break;
                case ConversationFlow.Question.Height:
                    if (ValidateHeight(input, out var height, out message))
                    {
                        profile.height = height;
                        await turnContext.SendActivityAsync($"OK. Your height {profile.height} is saved.", null, null, cancellationToken);
                        await turnContext.SendActivityAsync("What is your weight?", null, null, cancellationToken);
                        flow.LastQuestionAsked = ConversationFlow.Question.Weight;
                        break;
                    }
                    else
                    {
                        await turnContext.SendActivityAsync(message ?? "I'm sorry, I didn't understand that.", null, null, cancellationToken);
                        break;
                    }
                case ConversationFlow.Question.Weight:
                    if (ValidateWeight(input, out var weight, out message))
                    {
                        profile.weight = weight;
                        await turnContext.SendActivityAsync($"I have your weight as {profile.weight}.", null, null, cancellationToken);
                        await turnContext.SendActivityAsync("How old are you?", null, null, cancellationToken);
                        flow.LastQuestionAsked = ConversationFlow.Question.Age;
                        break;
                    }
                    else
                    {
                        await turnContext.SendActivityAsync(message ?? "I'm sorry, I didn't understand that.", null, null, cancellationToken);
                        break;
                    }

                case ConversationFlow.Question.Age:
                    if (ValidateAge(input, out var age, out message))
                    {
                        profile.age = age;
                        await turnContext.SendActivityAsync($"I have your age as {profile.age}.");
                        await turnContext.SendActivityAsync($"What's your gender?");
                        flow.LastQuestionAsked = ConversationFlow.Question.Sex;
                        break;
                    }
                    else
                    {
                        await turnContext.SendActivityAsync(message ?? "I'm sorry, I didn't understand that.", null, null, cancellationToken);
                        break;
                    }
                
                case ConversationFlow.Question.Sex:
                    if (ValidateSex(input, out string sex, out message))
                    {
                        profile.sex = sex;
                        await turnContext.SendActivityAsync($"Your gender: {profile.sex}.");
                        await turnContext.SendActivityAsync($"Evaluate your physical activity (1.2 - passive lifestyle, 1.9 - active sports)");
                        flow.LastQuestionAsked = ConversationFlow.Question.PhysActCoef;
                        break;
                    }
                    else
                    {
                        await turnContext.SendActivityAsync(message ?? "I'm sorry, I didn't understand that.", null, null, cancellationToken);
                        break;
                    }

                case ConversationFlow.Question.PhysActCoef:
                    if (ValidatePhysActCoef(input, out var phys_act_coef, out message))
                    {
                        profile.phys_act_coef = phys_act_coef;
                        await turnContext.SendActivityAsync($"I remember your weight as {profile.phys_act_coef}.");

                        string localPrefix = Path.GetTempPath();
                
                        PersonalDataRecord rec = profile;
                        rec.ComputeNorms();

                        string userId = turnContext.Activity.From.Id;

                        string eatingCsvLocal = Path.Combine(localPrefix, userId + "eat.csv");
                        string eatingCsvBlob = Path.Combine(userId, "eat.csv");

                        CSVHandler.WriteToCsv(new List<EatingRecord>(), eatingCsvLocal);
                        bsh.Upload(eatingCsvLocal, eatingCsvBlob).Wait();

                        string persCsvLocal = Path.Combine(localPrefix, userId + "pers.csv");
                        string persCsvBlob = Path.Combine(userId, "pers.csv");

                        CSVHandler.WriteToCsv(new List<PersonalDataRecord>{rec}, persCsvLocal);
                        bsh.Upload(persCsvLocal, persCsvBlob).Wait();

                        await turnContext.SendActivityAsync($"Now you can use Diet Bot!");
                        flow.LastQuestionAsked = ConversationFlow.Question.Finished;
                        break;
                    }
                    else
                    {
                        await turnContext.SendActivityAsync(message ?? "I'm sorry, I didn't understand that.", null, null, cancellationToken);
                        break;
                    }
            }
        }

        private static bool ValidateHeight(string input, out float height, out string message)
        {
            height = float.Parse(input.Trim(), CultureInfo.InvariantCulture.NumberFormat);
            message = null;

            if (height < 20.0f || height > 260.0f)
            {
                message = "Recheck your height input please.";
            }

            return message is null;
        }

        private static bool ValidateWeight(string input, out float weight, out string message)
        {
            weight = float.Parse(input.Trim(), CultureInfo.InvariantCulture.NumberFormat);
            message = null;

            if (weight < 20.0f || weight > 260.0f)
            {
                message = "Recheck your weight input please.";
            }

            return message is null;
        }

        private static bool ValidateAge(string input, out int age, out string message)
        {
            age = Int32.Parse(input.Trim());
            message = null;

            if (age < 16 || age > 120)
            {
                message = "Recheck your age input please.";
            }

            return message is null;
        }

        private static bool ValidateSex(string input, out string sex, out string message)
        {
            sex = input.Trim();
            message = null;

            if (sex != "male" && sex != "female")
            {
                message = "Recheck your gender input please.";
            }

            return message is null;
        }

        private static bool ValidatePhysActCoef(string input, out float phys_act_coef, out string message)
        {
            phys_act_coef = float.Parse(input, CultureInfo.InvariantCulture.NumberFormat);
            message = null;

            if (phys_act_coef < 1.2f || phys_act_coef > 1.9f)
            {
                message = "Recheck your physical activity coefficient please.";
            }

            return message is null;
        }
    }
}