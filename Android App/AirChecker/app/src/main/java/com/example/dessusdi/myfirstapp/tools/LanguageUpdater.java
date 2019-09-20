package com.example.dessusdi.myfirstapp.tools;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.Configuration;

import com.example.dessusdi.myfirstapp.R;

import java.util.Locale;


public class LanguageUpdater {

    private SharedPreferences sharedPreferences;
    private Context context;

    /**
     * @param context context of the activity
     * @param sharedPreferences preferences of the app
     */
    public LanguageUpdater(Context context, SharedPreferences sharedPreferences) {
        this.context = context;
        this.sharedPreferences = sharedPreferences;
    }

    /**
     * Public method allowing to set dynamically language
     * @param newLanguage new language string ('fr', 'en' etc.).
     * @param recreateActivity boolean value to recreate or not activity
     */
    public void buildLanguageConfiguration(String newLanguage, boolean recreateActivity) {
        Locale locale = new Locale(newLanguage);
        Locale.setDefault(locale);
        Configuration config = new Configuration();
        config.locale = locale;
        this.refreshUI(config, recreateActivity);
    }

    /**
     * Loading saved language from preferences
     */
    public void loadSavedLanguage() {
        this.buildLanguageConfiguration(getSavedLocale(), false);
    }

    /**
     * @return language saved from preferences ('fr', 'en', etc.).
     */
    public String getSavedLocale() {
        return this.sharedPreferences.getString("language_preference", context.getResources().getString(R.string.language_english));
    }

    /**
     * Refresh current UI after setting language
     * @param config new configuration
     * @param recreateActivity boolean value, true to recreate the activity, false to ignore
     */
    private void refreshUI(Configuration config, boolean recreateActivity) {
        Activity activity = (Activity)context;
        activity.getResources().updateConfiguration(config, null);

        if(recreateActivity)
            activity.recreate();
    }

}
