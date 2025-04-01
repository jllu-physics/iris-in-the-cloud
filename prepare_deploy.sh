DEPLOY_ENVIRON="$1"
if [ "$DEPLOY_ENVIRON" = "dev" ]; then
    echo "preparing deployment to local environment"
    cp ./static/config.dev.js ./static/config.js
elif [ "$DEPLOY_ENVIRON" = "stage" ]; then
    echo "preparing deployment to stage environment"
    cp ./static/config.stage.js ./static/config.js
else
    echo "Unknown environment"
fi


# design tradeoff:
#
# It seems inconsistent that only the config for
# index.html is changed here, and the config for
# FastAPI is changed in dockerfile.
#
# The idea, however, is that the front end and
# backend should not have been sharing environment
# in the first place and the minimal front end
# is just here to make the inference endpoint
# accessible to human directly.
#
# If we just want an API, we don't need front end
# If we really want a nice front end, I would leave
# this to the expert.
# Either way, I will keep the front end to
# minimal viable product : )