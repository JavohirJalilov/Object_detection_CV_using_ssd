from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters
import main
import object_detection
import matplotlib.pyplot as plt
import cv2
import os

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        text="Send a picture for prediction",
    )

def photo(update: Update, context: CallbackContext) -> None:
    bot = context.bot
    chat_id = update.message.chat_id
    photo_file = update.message.photo[-1].get_file()
    image = photo_file.download()
    image_np = cv2.imread(image)
    # image = cv2.resize(image, (300, 300))
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    predict_image = object_detection.predict(image_np)
    cv2.imwrite('predict_image.jpg', predict_image)
    #open image file
    predict_image = open('predict_image.jpg', 'rb')
    os.remove('predict_image.jpg')
    os.remove(image)
    # send image to telegram
    bot.send_photo(chat_id=chat_id, photo=predict_image)
    

updater = Updater('1924852364:AAGvyEudObx4dXiUkHc-MF_Eo_48cC7-7BU')

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, photo))

updater.start_polling()
updater.idle()